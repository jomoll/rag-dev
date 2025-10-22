import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Search, ShieldAlert, Clock, FileText, ChevronDown, Loader2, Copy, Check, Filter, Settings, Database, ListFilter, Plus, Download, Trash2, Info, Link as LinkIcon, ExternalLink, User, Lock, TimerReset, X, Users, Edit2, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetFooter, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT_NAME } from "@/api_keys";

// Updated Types to match database schema
export type Patient = { 
  id: string; 
  firstname?: string;
  lastname?: string;
  fullname: string; 
  dob?: string; 
};

// New type for patient groups
export type PatientGroup = {
  id: string;
  name: string;
  patients: Patient[];
  createdAt: Date;
};

export type DocHit = {
  id: string; // section_id or report_id
  patientId: string;
  patientName?: string; // Add patient name for group queries
  reportId?: string;
  docType: string; // report_type
  section?: string; // section_name
  date: string; // report_date (ISO)
  score?: number;
  snippet: string; // section_content or content
  url?: string;
  pinned?: boolean;
  wordCount?: number;
  filename?: string;
};

export type RetrievalSettings = {
  k: number;
  startDate?: Date | null;
  endDate?: Date | null;
  docTypes: string[]; // report types
  sectionTypes: string[]; // section types
  searchLevel: "sections" | "reports" | "both"; // new: what to search
  indexVersion?: string;
  // Add custom prompt settings
  useCustomPrompt: boolean;
  customPrompt: string;
  // Add fallback settings
  enableScoreFallback: boolean;
  minScoreThreshold: number;
};

export type RunMetadata = {
  modelId: string;
  promptTemplateHash: string;
  indexSnapshotId: string;
  latencyMs?: { retrieval?: number; generation?: number; total?: number };
};

export type RagAnswer = {
  status: "ok" | "insufficient" | "error";
  text: string;
  citedDocIds: string[];
  warnings?: string[];
};

// Configuration for local Llama model endpoint
const LLM_CONFIG = {
  baseUrl: "http://localhost:9999/v1",
  token: "dacbebe8c973154018a3d0f5",
  timeout: 120000,
  modelName: "llama-3.3-70b-instruct", // Adjust this to your actual model name
};

// Configuration for Azure OpenAI GPT-5 endpoint
const LLM_CONFIG2 = {
  azureEndpoint: ENDPOINT,
  apiKey: API_KEY,
  apiVersion: API_VERSION,
  deploymentName: DEPLOYMENT_NAME,
  timeout: 120000,
};

// Update the API configuration
const API_BASE_URL = 'http://10.32.16.225:5200/api';  // Always use server IP for network access

// Updated API layer to work with new database structure
const api = {
  async listPatients(q: string): Promise<Patient[]> {
    console.log('🔍 Searching for patients with query:', q);
    
    try {
      const response = await fetch(`${API_BASE_URL}/patients/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: q,
          searchFields: ['fullname', 'firstname', 'lastname', 'patient_id', 'dob'] // Enable multi-field search
        })
      });
      
      console.log('📡 API response status:', response.status);
      
      if (!response.ok) {
        console.warn('⚠️ API server error');
        return [];
      }
      
      const results = await response.json();
      console.log('✅ API results:', results);
      
      const patients = results.map((row: any) => ({
        id: row.patient_id,
        firstname: row.firstname,
        lastname: row.lastname,
        fullname: row.fullname || `${row.firstname || ''} ${row.lastname || ''}`.trim(),
        dob: row.dob
      }));
      
      console.log('👥 Mapped patients:', patients);
      return patients;
    } catch (error) {
      console.error('❌ Error searching patients:', error);
      return [];
    }
  },

  // New method to get available report types
  async getReportTypes(): Promise<string[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/reports/types`);
      if (response.ok) {
        const data = await response.json();
        return data.map((item: any) => item.report_type);
      }
    } catch (error) {
      console.error('❌ Error fetching report types:', error);
    }
    return ["Arztbrief", "Cytologie", "Flow cytometry"];
  },

  // New method to get available section types ordered by frequency
  async getSectionTypes(): Promise<string[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/sections/types`);
      if (response.ok) {
        const data = await response.json();
        // Assuming the API returns objects with section_name and count
        // Sort by count (most common first) instead of alphabetically
        return data
          .sort((a: any, b: any) => (b.count || 0) - (a.count || 0))
          .map((item: any) => item.section_name);
      }
    } catch (error) {
      console.error('❌ Error fetching section types:', error);
    }
    return [];
  },
  
  async runRag(params: {
    question: string;
    patientId?: string; // Make optional for group queries
    patientIds?: string[]; // Add for group queries
    settings: RetrievalSettings;
    pinnedDocIds: string[];
  }): Promise<{ hits: DocHit[]; answer: RagAnswer; meta: RunMetadata }> {
    const t0 = performance.now();
    
    const isGroupQuery = params.patientIds && params.patientIds.length > 0;
    const queryTarget = isGroupQuery ? `${params.patientIds.length} patients` : `patient ${params.patientId}`;
    
    console.log('🔍 RAG Search starting with params:', {
      question: params.question,
      patientId: params.patientId,
      patientIds: params.patientIds,
      isGroupQuery,
      k: params.settings.k,
      searchLevel: params.settings.searchLevel,
      docTypes: params.settings.docTypes,
      sectionTypes: params.settings.sectionTypes
    });
    
    let hits: DocHit[] = [];
    
    try {
      // Choose search endpoint based on search level
      let searchEndpoint = `${API_BASE_URL}/search`;
      
      if (params.settings.searchLevel === "sections") {
        searchEndpoint = `${API_BASE_URL}/sections/search`;
      } else if (params.settings.searchLevel === "reports") {
        searchEndpoint = `${API_BASE_URL}/reports/search`;
      }
      
      // For group queries, multiply k by number of patients to get k results per patient
      const effectiveK = isGroupQuery ? params.settings.k * params.patientIds!.length : params.settings.k;
      
      const searchPayload = {
        query: params.question,
        patient_id: isGroupQuery ? undefined : params.patientId,
        patient_ids: isGroupQuery ? params.patientIds : undefined, // Add group support
        k: effectiveK, // Use effective k for group queries
        k_per_patient: isGroupQuery ? params.settings.k : undefined, // Add k per patient for backend
        report_types: params.settings.docTypes.length > 0 ? params.settings.docTypes : null,
        section_types: params.settings.sectionTypes.length > 0 ? params.settings.sectionTypes : null,
        // Fix timezone issue by using local date formatting instead of toISOString()
        start_date: params.settings.startDate ? 
          `${params.settings.startDate.getFullYear()}-${String(params.settings.startDate.getMonth() + 1).padStart(2, '0')}-${String(params.settings.startDate.getDate()).padStart(2, '0')}` : 
          undefined,
        end_date: params.settings.endDate ? 
          `${params.settings.endDate.getFullYear()}-${String(params.settings.endDate.getMonth() + 1).padStart(2, '0')}-${String(params.settings.endDate.getDate()).padStart(2, '0')}` : 
          undefined,
        search_level: params.settings.searchLevel,
        // Add fallback parameters
        enable_score_fallback: params.settings.enableScoreFallback,
        min_score_threshold: params.settings.minScoreThreshold
      };
      
      console.log('📡 Sending search request to:', searchEndpoint);
      console.log('📡 Search payload:', searchPayload);
      
      const response = await fetch(searchEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchPayload)
      });
      
      console.log('📡 Search response status:', response.status);
      
      if (response.ok) {
        const searchResults = await response.json();
        console.log('✅ Search results:', searchResults);
        console.log('📊 Found', searchResults.length, 'documents/sections across', queryTarget);
        
        // Convert to DocHit format (works for both sections and reports)
        hits = searchResults.map((row: any) => ({
          id: row.section_id || row.report_id,
          patientId: row.patient_id,
          patientName: row.patient_name || row.fullname, // Include patient name for group queries
          reportId: row.report_id,
          docType: row.report_type,
          section: row.section_name,
          date: row.report_date,
          score: row.score,
          snippet: row.section_content || row.content,
          url: `#/document/${row.report_id}${row.section_id ? `/section/${row.section_id}` : ''}`,
          pinned: false,
          wordCount: row.word_count,
          filename: row.filename
        })).sort((a, b) => (b.score || 0) - (a.score || 0)); // Sort by score descending (highest first)
        
        console.log('🎯 Mapped and sorted document hits:', hits);
      } else {
        console.error('❌ Search failed with status:', response.status);
        const errorText = await response.text();
        console.error('❌ Error response:', errorText);
      }
    } catch (error) {
      console.error('❌ Error in search:', error);
    }
    
    const retrievalMs = performance.now() - t0;
    console.log('⏱️ Retrieval took:', retrievalMs, 'ms');
    
    // Generate answer using Llama model via API proxy
    const generationStart = performance.now();
    let answer: RagAnswer;
    
    try {
      // Use API server as proxy instead of direct connection
      const url = `${API_BASE_URL}/llama/chat`;
      
      // Build context from retrieved documents/sections
      const context = hits.map(hit => {
        const patientInfo = isGroupQuery && hit.patientName ? `Patient: ${hit.patientName} | ` : '';
        const header = `[${patientInfo}${hit.docType}${hit.section ? ` - ${hit.section}` : ''} | ${hit.date?.substring(0, 10)} | Score: ${hit.score?.toFixed(2)}${hit.filename ? ` | ${hit.filename}` : ''}]`;
        return `${header}\n${hit.snippet}`;
      }).join('\n\n---\n\n');
      
      console.log('📝 Built context for LLM:', context.length, 'characters');
      console.log('📄 Context preview:', context.substring(0, 200) + '...');
      
      // Modify system prompt for group queries
      const baseSystemPrompt = params.settings.useCustomPrompt 
        ? params.settings.customPrompt.replace('{context}', context)
        : `Du bist ein klinischer KI-Assistent, der deutsche medizinische Berichte analysiert. Beantworte Fragen basierend AUSSCHLIESSLICH auf dem bereitgestellten medizinischen Kontext. Wenn der Kontext nicht genügend Informationen zur Beantwortung der Frage enthält, gib das klar an.

Beim Zitieren von Informationen erwähne:
- Den Dokumenttyp (Arztbrief, Cytologie, Flow cytometry)
- Den Abschnittsnamen, wenn verfügbar
- Das Datum des Dokuments
${isGroupQuery ? '- Den Patientennamen, um zwischen verschiedenen Patienten zu unterscheiden' : ''}
- Sei spezifisch darüber, welches Dokument die relevanten Informationen enthält

${isGroupQuery ? 'Du analysierst Daten von mehreren Patienten gleichzeitig. Achte darauf, die Informationen klar nach Patienten zu trennen und Vergleiche oder Gemeinsamkeiten zwischen den Patienten hervorzuheben, wenn relevant.' : ''}

Antworte auf Deutsch, auch wenn die Frage auf Englisch gestellt wird.

MEDIZINISCHE AKTEN KONTEXT:
${context}`;

      const systemPrompt = context.length > 0 
        ? baseSystemPrompt
        : (params.settings.useCustomPrompt
            ? params.settings.customPrompt.replace('{context}', 'Es sind keine spezifischen Patientendaten verfügbar.')
            : `Du bist ein klinischer KI-Assistent. Es sind keine spezifischen Patientendaten verfügbar. Gib eine allgemeine medizinische Antwort und weise klar auf das Fehlen patientenspezifischer Informationen hin.`);

      console.log('🤖 Using system prompt with context length:', systemPrompt.length);

      const requestBody = {
        model: LLM_CONFIG.modelName,
        messages: [
          {
            role: "system",
            content: systemPrompt
          },
          {
            role: "user",
            content: params.question
          }
        ],
        temperature: 0.1,
        max_tokens: 2048,
        stream: false
      };

      // Retry configuration
      const maxRetries = 10000;
      const retryDelay = 1000; // 1 second constant backoff
      
      let lastError: any;
      
      for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
          console.log(`🚀 Llama attempt ${attempt + 1}/${maxRetries + 1}:`, url);
          
          const response = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
          });

          console.log('📡 API proxy response status:', response.status);

          if (response.ok) {
            const data = await response.json();
            console.log('🦙 Llama response via proxy successful');
            
            const generatedText = data?.choices?.[0]?.message?.content || "Keine Antwort generiert.";
            
            console.log('📝 Extracted text:', generatedText);
            
            answer = {
              status: hits.length > 0 ? "ok" : "insufficient",
              text: generatedText,
              citedDocIds: hits.map(h => h.id),
              warnings: hits.length === 0 ? [`Keine relevanten Dokumente in den ${isGroupQuery ? 'Patientengruppen-' : 'Patienten'}akten gefunden`] : undefined
            };
            
            break; // Success - exit retry loop
            
          } else {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            lastError = new Error(`API proxy error: ${response.status} - ${errorData.error || 'Unknown error'}`);
            
            // Check if we should retry based on status code
            const shouldRetry = response.status >= 500 || response.status === 429 || response.status === 503;
            
            if (!shouldRetry || attempt === maxRetries) {
              console.error(`❌ Non-retryable error or max retries reached:`, lastError);
              throw lastError;
            }
            
            console.log(`⏳ Llama busy (${response.status}), retrying in ${retryDelay}ms... (attempt ${attempt + 1}/${maxRetries + 1})`);
            await new Promise(resolve => setTimeout(resolve, retryDelay));
          }
          
        } catch (fetchError: any) {
          lastError = fetchError;
          
          // Check if it's a network error that might be retryable
          const isNetworkError = fetchError.name === 'TypeError' || fetchError.message?.includes('fetch');
          
          if (!isNetworkError || attempt === maxRetries) {
            console.error(`❌ Network error or max retries reached:`, fetchError);
            break;
          }
          
          console.log(`⏳ Network error, retrying in ${retryDelay}ms... (attempt ${attempt + 1}/${maxRetries + 1})`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
      }
      
      // If we get here without setting answer, all retries failed
      if (!answer) {
        throw lastError || new Error('All retry attempts failed');
      }
      
    } catch (error: any) {
      console.error('❌ Llama generation error:', error);
      
      let errorMessage = "Unbekannter Fehler beim Generieren der Antwort";
      
      if (error.message?.includes('504')) {
        errorMessage = "Timeout beim Llama-Modell auf dem Cluster";
      } else if (error.message?.includes('503') || error.message?.includes('502')) {
        errorMessage = "Llama-Server auf dem Cluster überlastet - bitte versuchen Sie es in ein paar Minuten erneut";
      } else if (error.message?.includes('429')) {
        errorMessage = "Llama-Server ist überlastet - zu viele Anfragen";
      } else if (error.message?.includes('500')) {
        errorMessage = "Interner Fehler im Llama-Server";
      } else {
        errorMessage = `Llama-Fehler: ${error.message}`;
      }
      
      answer = {
        status: "error",
        text: errorMessage,
        citedDocIds: [],
        warnings: [
          "Verbindung zum Llama-Modell über API-Proxy fehlgeschlagen",
          `Proxy: ${API_BASE_URL}/llama/chat`,
          `Modell: ${LLM_CONFIG.modelName}`,
          "Versucht automatisch 10x mit 1s Verzögerung"
        ]
      };
    }
    
    const generationMs = performance.now() - generationStart;
    const meta: RunMetadata = {
      modelId: LLM_CONFIG.modelName,
      promptTemplateHash: isGroupQuery ? "clinical-german-group-v1" : "clinical-german-v1",
      indexSnapshotId: `sqlite-fts-${params.settings.searchLevel}-${params.settings.k}${isGroupQuery ? '-group' : ''}`,
      latencyMs: { 
        retrieval: Math.round(retrievalMs), 
        generation: Math.round(generationMs), 
        total: Math.round(retrievalMs + generationMs) 
      },
    };
    
    return { hits, answer, meta };
  },

  async getFullReport(reportId: string): Promise<{
    content: string;
    filename?: string;
    docType: string;
    date: string;
    patientId: string;
  } | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/reports/${reportId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          content: data.content,
          filename: data.filename,
          docType: data.report_type,
          date: data.report_date,
          patientId: data.patient_id
        };
      }
    } catch (error) {
      console.error('❌ Error fetching full report:', error);
    }
    return null;
  }
};

// Utility functions
function delay(ms: number) {
  return new Promise(res => setTimeout(res, ms));
}

function formatDate(iso: string) {
  if (!iso) return "Unknown date";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
}

// Missing helper components
function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  );
}

function ToggleChip({ 
  label, 
  active, 
  onToggle 
}: { 
  label: string; 
  active: boolean; 
  onToggle: () => void; 
}) {
  return (
    <button
      onClick={onToggle}
      className={`px-2 py-1 text-xs rounded-full border transition-colors ${
        active 
          ? 'bg-blue-100 border-blue-300 text-blue-800' 
          : 'bg-gray-100 border-gray-300 text-gray-700 hover:bg-gray-200'
      }`}
    >
      {label}
    </button>
  );
}

function DateField({ 
  label, 
  value, 
  onChange 
}: { 
  label: string; 
  value: Date | null | undefined; 
  onChange: (date: Date | null) => void; 
}) {
  const formatDateToGerman = (date: Date | null | undefined): string => {
    if (!date) return "";
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${day}.${month}.${year}`;
  };

  const parseDateFromGerman = (dateStr: string): Date | null => {
    if (!dateStr.trim()) return null;
    
    const parts = dateStr.split('.');
    if (parts.length !== 3) return null;
    
    const day = parseInt(parts[0], 10);
    const month = parseInt(parts[1], 10) - 1; // Month is 0-indexed
    const year = parseInt(parts[2], 10);
    
    if (isNaN(day) || isNaN(month) || isNaN(year)) return null;
    if (day < 1 || day > 31 || month < 0 || month > 11 || year < 1900) return null;
    
    const date = new Date(year, month, day);
    // Verify the date is valid (handles cases like 31.02.2023)
    if (date.getDate() !== day || date.getMonth() !== month || date.getFullYear() !== year) {
      return null;
    }
    
    return date;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value;
    const parsedDate = parseDateFromGerman(inputValue);
    onChange(parsedDate);
  };

  return (
    <div>
      <Label>{label}</Label>
      <Input
        type="text"
        placeholder="DD.MM.YYYY"
        value={formatDateToGerman(value)}
        onChange={handleInputChange}
        className="mt-1"
      />
    </div>
  );
}

// Root component
export default function ClinicalRagApp() {
  const [patientQuery, setPatientQuery] = useState("");
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [selectedGroup, setSelectedGroup] = useState<PatientGroup | null>(null);
  const [patientGroups, setPatientGroups] = useState<PatientGroup[]>([]);
  const [showGroupBuilder, setShowGroupBuilder] = useState(false);
  const [reportTypes, setReportTypes] = useState<string[]>([]);
  const [sectionTypes, setSectionTypes] = useState<string[]>([]);

  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState<RetrievalSettings>({
    k: 8,
    startDate: undefined,
    endDate: undefined,
    docTypes: [],
    sectionTypes: [],
    searchLevel: "sections", // Default to searching sections
    indexVersion: "sections-v1",
    // Add custom prompt defaults
    useCustomPrompt: false,
    customPrompt: `Du bist ein klinischer KI-Assistent, der deutsche Berichte analysiert. Beantworte Fragen basierend AUSSCHLIESSLICH auf dem bereitgestellten medizinischen Kontext. Wenn der Kontext nicht genügend Informationen zur Beantwortung der Frage enthält, gib das klar an.

Beim Zitieren von Informationen erwähne:
- Den Dokumenttyp (Arztbrief, Cytologie, Flow cytometry)
- Den Abschnittsnamen, wenn verfügbar
- Das Datum des Dokuments
- Sei spezifisch darüber, welches Dokument die relevanten Informationen enthält

Antworte auf Deutsch, auch wenn die Frage auf Englisch gestellt wird.

MEDIZINISCHE AKTEN KONTEXT:
{context}`,
    // Add fallback defaults
    enableScoreFallback: false,
    minScoreThreshold: 0.5, // Default threshold of 0.5
  });

  const [isRunning, setIsRunning] = useState(false);
  const [hits, setHits] = useState<DocHit[]>([]);
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [meta, setMeta] = useState<RunMetadata | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedReport, setSelectedReport] = useState<{
    reportId: string;
    patientId: string;
    filename?: string;
    docType: string;
    date: string;
    content: string;
    highlightedContent: string;
    highlightSectionId?: string;
    highlightSection?: string;
  } | null>(null);

  const [loggingConfig, setLoggingConfig] = useState<{
    enabled: boolean;
    filepath: string;
    format: 'json' | 'csv';
    includeFullContent: boolean;
    autoLog?: boolean; // Add autoLog property to the type
  }>({
    enabled: false,
    filepath: '/data/moll/interagt/logs/test.log', // Set default log file path
    format: 'json',
    includeFullContent: false,
    autoLog: true // Set default auto-log to true
  });

  const [isLogging, setIsLogging] = useState(false);
  const [logContent, setLogContent] = useState<string>('');
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [logCopied, setLogCopied] = useState(false);

  // Add these state variables to the component
  const [groupName, setGroupName] = useState("");
  const [selectedPatientIds, setSelectedPatientIds] = useState<string[]>([]);

  // Load metadata on mount
  useEffect(() => {
    console.log('🚀 Component mounted, loading metadata');
    
    // Load all patients
    api.listPatients('').then(list => {
      console.log('🏥 Initial patients loaded:', list);
      setPatients(list);
    });
    
    // Load report types
    api.getReportTypes().then(types => {
      console.log('📋 Report types loaded:', types);
      setReportTypes(types);
    });
    
    // Load section types  
    api.getSectionTypes().then(types => {
      console.log('📑 Section types loaded:', types);
      setSectionTypes(types);
    });

    // Load saved groups from localStorage
    const savedGroups = localStorage.getItem('patientGroups');
    if (savedGroups) {
      try {
        const parsed = JSON.parse(savedGroups);
        const groupsWithDates = parsed.map((g: any) => ({
          ...g,
          createdAt: new Date(g.createdAt)
        }));
        setPatientGroups(groupsWithDates);
        console.log('📥 Loaded saved groups:', groupsWithDates);
      } catch (error) {
        console.error('❌ Error loading saved groups:', error);
      }
    }
  }, []);

  // Save groups to localStorage whenever they change
  useEffect(() => {
    if (patientGroups.length > 0) {
      localStorage.setItem('patientGroups', JSON.stringify(patientGroups));
      console.log('💾 Saved groups to localStorage');
    }
  }, [patientGroups]);

  // Search patients with debouncing
  useEffect(() => {
    let active = true;
    console.log('🔄 Patient search effect triggered, query:', patientQuery);
    
    const searchQuery = patientQuery.trim();
    
    // If query is empty, show all patients
    api.listPatients(searchQuery).then(list => {
      if (active) {
        console.log('📋 Setting patients state:', list);
        setPatients(list);
      }
    });
    
    return () => {
      active = false;
    };
  }, [patientQuery]);

  const pinnedIds = useMemo(() => hits.filter(h => h.pinned).map(h => h.id), [hits]);

  const canLog = useMemo(() => {
    return (selectedPatient || selectedGroup) && question.trim() && answer;
  }, [selectedPatient, selectedGroup, question, answer]);

  // Function to fetch log content
  async function fetchLogContent() {
    if (!loggingConfig.enabled || !loggingConfig.filepath) {
      setLogContent('');
      return;
    }

    setIsLoadingLogs(true);
    try {
      const response = await fetch(`${API_BASE_URL}/logging/content`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filepath: loggingConfig.filepath,
          format: loggingConfig.format
        })
      });

      if (response.ok) {
        const data = await response.json();
        setLogContent(data.content || '');
      } else {
        setLogContent('Error loading log file content');
      }
    } catch (error) {
      console.error('❌ Error fetching log content:', error);
      setLogContent('Error loading log file content');
    } finally {
      setIsLoadingLogs(false);
    }
  }

  // Auto-refresh log content when logging is enabled
  useEffect(() => {
    if (loggingConfig.enabled && loggingConfig.filepath) {
      fetchLogContent();
      const interval = setInterval(fetchLogContent, 5000); // Refresh every 5 seconds
      return () => clearInterval(interval);
    } else {
      setLogContent('');
    }
  }, [loggingConfig.enabled, loggingConfig.filepath, loggingConfig.format]);

  // Refresh logs after successful logging
  useEffect(() => {
    if (loggingConfig.enabled && loggingConfig.filepath && answer) {
      // Small delay to ensure log is written
      setTimeout(fetchLogContent, 1000);
    }
  }, [answer, loggingConfig.enabled, loggingConfig.filepath]);

  async function run() {
    if (!selectedPatient && !selectedGroup) return; // Guard clause - patient or group is required
    
    setIsRunning(true);
    try {
      const runParams = selectedGroup 
        ? {
            question: question.trim(),
            patientIds: selectedGroup.patients.map(p => p.id),
            settings,
            pinnedDocIds: pinnedIds,
          }
        : {
            question: question.trim(),
            patientId: selectedPatient!.id,
            settings,
            pinnedDocIds: pinnedIds,
          };
      
      const res = await api.runRag(runParams);
      setHits(res.hits);
      setAnswer(res.answer);
      setMeta(res.meta);

      // Auto-log if logging is enabled
      if (loggingConfig.enabled && loggingConfig.filepath && res.answer) {
        try {
          await logCurrentSession();
          console.log('✅ Auto-logged session');
        } catch (logError) {
          console.warn('⚠️ Auto-logging failed:', logError);
        }
      }
    } finally {
      setIsRunning(false);
    }
  }

  function togglePin(id: string) {
    setHits(prev => prev.map(h => (h.id === id ? { ...h, pinned: !h.pinned } : h)));
  }

  function clearAll() {
    setQuestion("");
    setHits([]);
    setAnswer(null);
    setMeta(null);
  }

  async function copyWithCitations() {
    const cite = hits
      .filter(h => answer?.citedDocIds.includes(h.id))
      .map(h => `[${h.patientName ? `${h.patientName} • ` : ''}${h.docType}${h.section ? ` - ${h.section}` : ''} • ${formatDate(h.date)}]`)
      .join("; ");
    const payload = `${answer?.text || ""}\n\nSources: ${cite}`.trim();
    await navigator.clipboard.writeText(payload);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  const canRun = !!(selectedPatient || selectedGroup); // Check if patient or group is selected

  async function openReport(hit: DocHit) {
    if (!hit.reportId) return;
    
    try {
      const reportData = await api.getFullReport(hit.reportId);
      if (reportData) {
        let highlightedContent = reportData.content;
        
        // If we have section content from the search hit, highlight it directly
        if (hit.snippet && hit.snippet.trim().length > 20) {
          // Use the actual section content from the search hit
          const snippetToHighlight = hit.snippet.trim();
          
          // Find this exact content in the full report and highlight it
          if (reportData.content.includes(snippetToHighlight)) {
            const escapedSnippet = snippetToHighlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            highlightedContent = reportData.content.replace(
              new RegExp(escapedSnippet, 'g'),
              `<div style="background-color: #fef3c7; padding: 8px; border-left: 4px solid #f59e0b; margin: 4px 0; border-radius: 4px;">${snippetToHighlight}</div>`
            );
          } else {
            // Fallback: try to find a substantial portion of the snippet
            const words = snippetToHighlight.split(/\s+/);
            if (words.length >= 5) {
              // Try with first 50-100 characters of meaningful content
              const partialSnippet = words.slice(0, Math.min(10, words.length)).join(' ');
              if (reportData.content.includes(partialSnippet)) {
                const escapedPartial = partialSnippet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                highlightedContent = reportData.content.replace(
                  new RegExp(escapedPartial, 'g'),
                  `<mark style="background-color: #fef3c7; padding: 2px 4px; border-radius: 2px;">${partialSnippet}</mark>`
                );
              }
            }
          }
        }
        
        // Convert newlines to HTML for proper rendering
        highlightedContent = highlightedContent.replace(/\n/g, '<br>');
        
        setSelectedReport({
          reportId: hit.reportId,
          patientId: hit.patientId,
          filename: hit.filename || reportData.filename,
          docType: hit.docType,
          date: hit.date,
          content: reportData.content,
          highlightedContent: highlightedContent,
          highlightSectionId: hit.id,
          highlightSection: hit.section
        });
      }
    } catch (error) {
      console.error('❌ Error opening report:', error);
    }
  }

  async function logCurrentSession() {
    if ((!selectedPatient && !selectedGroup) || !question || !answer) {
      console.warn('⚠️ Cannot log: missing patient/group, question, or answer');
      return;
    }

    setIsLogging(true);
    
    try {
      const sessionId = selectedGroup 
        ? `group_${selectedGroup.id}_${Date.now()}`
        : `${selectedPatient!.id}_${Date.now()}`;

      const logEntry = {
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        query_type: selectedGroup ? 'group' : 'individual',
        patient: selectedGroup ? undefined : {
          id: selectedPatient!.id,
          name: selectedPatient!.fullname,
          dob: selectedPatient!.dob
        },
        group: selectedGroup ? {
          id: selectedGroup.id,
          name: selectedGroup.name,
          patient_count: selectedGroup.patients.length,
          patients: selectedGroup.patients.map(p => ({
            id: p.id,
            name: p.fullname,
            dob: p.dob
          }))
        } : undefined,
        query: {
          question: question.trim(),
          settings: {
            k: settings.k,
            searchLevel: settings.searchLevel,
            docTypes: settings.docTypes,
            sectionTypes: settings.sectionTypes,
            dateRange: {
              start: settings.startDate?.toISOString().split('T')[0],
              end: settings.endDate?.toISOString().split('T')[0]
            }
          }
        },
        results: {
          answer: {
            status: answer.status,
            text: answer.text,
            warnings: answer.warnings
          },
          sources: hits.map(hit => ({
            id: hit.id,
            reportId: hit.reportId,
            patientId: hit.patientId,
            patientName: hit.patientName,
            docType: hit.docType,
            section: hit.section,
            date: hit.date,
            score: hit.score,
            wordCount: hit.wordCount,
            filename: hit.filename,
            snippet: loggingConfig.includeFullContent ? hit.snippet : hit.snippet?.substring(0, 200) + '...'
          })),
          sourceCount: hits.length,
          patientDistribution: selectedGroup ? 
            hits.reduce((acc, hit) => {
              acc[hit.patientId] = (acc[hit.patientId] || 0) + 1;
              return acc;
            }, {} as Record<string, number>) : undefined
        },
        metadata: {
          modelId: meta?.modelId,
          latency: meta?.latencyMs,
          indexVersion: meta?.indexSnapshotId
        }
      };

      // Send to logging API
      const response = await fetch(`${API_BASE_URL}/logging/session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...logEntry,
          filepath: loggingConfig.filepath,
          format: loggingConfig.format
        })
      });

      if (response.ok) {
        console.log('✅ Session logged successfully');
        // Show success notification
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded z-50';
        notification.textContent = `Logged to ${loggingConfig.filepath}`;
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 3000);
      } else {
        throw new Error(`Logging failed: ${response.status}`);
      }
      
    } catch (error) {
      console.error('❌ Logging error:', error);
      // Show error notification
      const notification = document.createElement('div');
      notification.className = 'fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded z-50';
      notification.textContent = 'Logging failed - check console';
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } finally {
      setIsLogging(false);
    }
  }

  // Add this function inside the component
  async function copyLogContent() {
    try {
      if (!logContent || logContent.trim() === '') {
        console.warn('⚠️ No log content to copy');
        return;
      }
      
      // Try the modern clipboard API first
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(logContent);
        console.log('✅ Log content copied using modern clipboard API');
      } else {
        // Fallback to older method
        const textArea = document.createElement('textarea');
        textArea.value = logContent;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        
        if (!successful) {
          throw new Error('Fallback copy method failed');
        }
        console.log('✅ Log content copied using fallback method');
      }
      
      setLogCopied(true);
      setTimeout(() => setLogCopied(false), 1200);
      
      // Show success notification
      const notification = document.createElement('div');
      notification.className = 'fixed top-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded z-50 text-sm';
      notification.textContent = `Copied ${logContent.length} characters to clipboard`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 2000);
      
    } catch (error) {
      console.error('❌ Error copying log content:', error);
      setLogCopied(false);
      
      // Show error notification with more details
      const notification = document.createElement('div');
      notification.className = 'fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded z-50 text-sm';
      notification.textContent = `Copy failed: ${error instanceof Error ? error.message : 'Unknown error'}`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    }
  }

  function handlePatientSelect(patient: Patient) {
    setSelectedPatient(patient);
    setSelectedGroup(null); // Clear group selection
  }

  function handleGroupSelect(group: PatientGroup) {
    setSelectedGroup(group);
    setSelectedPatient(null); // Clear individual patient selection
  }

  function deleteGroup(groupId: string) {
    setPatientGroups(prev => prev.filter(g => g.id !== groupId));
    if (selectedGroup?.id === groupId) {
      setSelectedGroup(null);
    }
  }

  // Add these missing functions:
  function togglePatientSelection(patientId: string) {
    setSelectedPatientIds(prev => 
      prev.includes(patientId)
        ? prev.filter(id => id !== patientId)
        : [...prev, patientId]
    );
  }

  function createGroup() {
    if (!groupName.trim() || selectedPatientIds.length === 0) return;
    
    const selectedPatients = patients.filter(p => selectedPatientIds.includes(p.id));
    
    const newGroup: PatientGroup = {
      id: `group_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: groupName.trim(),
      patients: selectedPatients,
      createdAt: new Date()
    };
    
    setPatientGroups(prev => [...prev, newGroup]);
    
    // Reset form and close modal
    setGroupName("");
    setSelectedPatientIds([]);
    setShowGroupBuilder(false);
    
    console.log('✅ Created new group:', newGroup);
  }

  // Prompt presets for different clinical scenarios
  const PROMPT_PRESETS = {
    'default-german': {
      name: 'Default (German)',
      description: 'Standard clinical assistant in German',
      prompt: `Du bist ein klinischer KI-Assistent, der deutsche medizinische Berichte analysiert. Beantworte Fragen basierend AUSSCHLIESSLICH auf dem bereitgestellten medizinischen Kontext. Wenn der Kontext nicht genügend Informationen zur Beantwortung der Frage enthält, gib das klar an.

Beim Zitieren von Informationen erwähne:
- Den Dokumenttyp (Arztbrief, Cytologie, Flow cytometry)
- Den Abschnittsnamen, wenn verfügbar
- Das Datum des Dokuments
- Sei spezifisch darüber, welches Dokument die relevanten Informationen enthält

Antworte auf Deutsch, auch wenn die Frage auf Englisch gestellt wird.

MEDIZINISCHE AKTEN KONTEXT:
{context}`
    },
    'english': {
      name: 'English Clinical',
      description: 'Clinical assistant responding in English',
      prompt: `You are a clinical AI assistant analyzing German medical reports. Answer questions based EXCLUSIVELY on the provided medical context. If the context doesn't contain sufficient information to answer the question, clearly state this.

When citing information, mention:
- The document type (Arztbrief, Cytologie, Flow cytometry)
- The section name, if available
- The document date
- Be specific about which document contains the relevant information

Respond in English, providing clear and concise medical information.

MEDICAL RECORDS CONTEXT:
{context}`
    },
    'structured-summary': {
      name: 'Structured Summary',
      description: 'Creates structured medical summaries in German',
      prompt: `Du bist ein medizinischer Datenanalyst. Analysiere die bereitgestellten Patientendaten und erstelle eine strukturierte Zusammenfassung im folgenden Format:

DIAGNOSEN:
- [Liste der erwähnten Diagnosen]

MEDIKAMENTE:
- [Liste der erwähnten Medikamente mit Dosierung wenn verfügbar]

BEFUNDE:
- [Wichtige Laborwerte und Untersuchungsergebnisse]

VERLAUF:
- [Chronologische Entwicklung wenn erkennbar]

Falls Informationen nicht im Kontext verfügbar sind, schreibe "Nicht dokumentiert".

MEDIZINISCHE AKTEN KONTEXT:
{context}`
    },
    'research-extraction': {
      name: 'Research Data Extraction',
      description: 'Extracts specific data points for research',
      prompt: `Du bist ein medizinischer Forscher. Extrahiere systematisch folgende Informationen aus den Patientendaten:

PATIENT: [Alter, Geschlecht wenn erwähnt]
HAUPTDIAGNOSE: [Primäre Diagnose]
NEBENDIAGNOSEN: [Sekundäre Diagnosen]
THERAPIE: [Behandlungen und Medikamente]
OUTCOME: [Behandlungsergebnis/Verlauf]
KOMPLIKATIONEN: [Aufgetretene Probleme]

Antworte nur mit den extrahierten Daten. Falls eine Information nicht verfügbar ist, schreibe "N/A".

MEDIZINISCHE AKTEN KONTEXT:
{context}`
    },
    'timeline': {
      name: 'Timeline Analysis',
      description: 'Creates chronological medical timelines',
      prompt: `Du bist ein medizinischer Zeitlinien-Analyst. Erstelle eine chronologische Übersicht der medizinischen Ereignisse basierend auf den bereitgestellten Dokumenten.

Format:
DATUM | EREIGNIS | QUELLE
[Datum] | [Was passierte] | [Dokumenttyp + Abschnitt]

Sortiere chronologisch (älteste zuerst). Falls Daten unklar sind, verwende "ca. [Datum]" oder "Unbekannt".

MEDIZINISCHE AKTEN KONTEXT:
{context}`
    },
    'group-comparison': {
      name: 'Group Comparison',
      description: 'Compares findings across multiple patients',
      prompt: `Du bist ein medizinischer Vergleichsanalyst. Analysiere die Daten mehrerer Patienten und erstelle einen strukturierten Vergleich:

GEMEINSAME BEFUNDE:
- [Was alle oder die meisten Patienten teilen]

UNTERSCHIEDE:
- [Wesentliche Unterschiede zwischen den Patienten]

PATIENTENSPEZIFISCHE HIGHLIGHTS:
[Für jeden Patienten die wichtigsten Besonderheiten]

KLINISCHE IMPLIKATIONEN:
- [Was bedeuten die Gemeinsamkeiten/Unterschiede]

Erwähne immer den Patientennamen bei spezifischen Befunden.

MEDIZINISCHE AKTEN KONTEXT:
{context}`
    }
  };

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gray-50">
        <header className="sticky top-0 z-30 backdrop-blur bg-white/80 border-b">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
            <Database className="w-5 h-5" />
            <h1 className="text-lg font-semibold">Clinical RAG System for Myeloma Patients</h1>
            <div className="ml-auto flex items-center gap-2">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2"><Info className="w-4 h-4" /> Details</Button>
                </SheetTrigger>
                <SheetContent side="right" className="w-[420px]">
                  <SheetHeader>
                    <SheetTitle>Run details</SheetTitle>
                  </SheetHeader>
                  <div className="mt-4 space-y-3 text-sm">
                    <div className="grid grid-cols-2 gap-3">
                      <Detail label="Model" value={meta?.modelId || "—"} />
                      <Detail label="Prompt" value={meta?.promptTemplateHash || "—"} />
                      <Detail label="Index" value={meta?.indexSnapshotId || settings.indexVersion} />
                      <Detail label="k" value={String(settings.k)} />
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <Detail label="Retrieval ms" value={meta?.latencyMs?.retrieval?.toString() || "—"} />
                      <Detail label="Gen ms" value={meta?.latencyMs?.generation?.toString() || "—"} />
                      <Detail label="Total ms" value={meta?.latencyMs?.total?.toString() || "—"} />
                    </div>
                    <div className="mt-2">
                      <h4 className="text-xs font-medium uppercase text-gray-500">Audit fields</h4>
                      <ul className="mt-1 text-xs list-disc pl-4 space-y-1 text-gray-600">
                        <li>query_type: {selectedGroup ? `group (${selectedGroup.patients.length} patients)` : selectedPatient ? "individual" : "—"}</li>
                        <li>patient_id: {selectedPatient?.id || "—"}</li>
                        <li>group_id: {selectedGroup?.id || "—"}</li>
                        <li>search_level: {settings.searchLevel}</li>
                        <li>useCustomPrompt: {String(settings.useCustomPrompt)}</li>
                        <li>scoreFallback: {settings.enableScoreFallback ? `enabled (≥${settings.minScoreThreshold})` : "disabled"}</li>
                        <li>docTypes: {settings.docTypes.length ? settings.docTypes.join(", ") : "all"}</li>
                        <li>sectionTypes: {settings.sectionTypes.length ? settings.sectionTypes.join(", ") : "all"}</li>
                      </ul>
                    </div>
                  </div>
                  <SheetFooter className="mt-6">
                    <Button variant="secondary" onClick={() => window.print()} className="w-full"><Download className="w-4 h-4 mr-2" /> Export</Button>
                  </SheetFooter>
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-6 grid lg:grid-cols-5 gap-6">
          {/* Left column: controls */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Patient Selection</CardTitle>
                <CardDescription>Search by name, patient ID, or date of birth to select a patient or create a group.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <Label htmlFor="patient">Search Patient</Label>
                  <Input 
                    id="patient" 
                    placeholder="Enter name, ID, or DOB (DD.MM.YYYY)" 
                    value={patientQuery} 
                    onChange={e => setPatientQuery(e.target.value)} 
                  />
                  
                  {/* Build Group Button */}
                  <div className="mt-3">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => setShowGroupBuilder(true)}
                      className="w-full gap-2"
                    >
                      <Users className="w-4 h-4" />
                      Build Group
                    </Button>
                  </div>
                  
                  <div className="mt-2 max-h-48 border rounded-md bg-white overflow-auto">
                    {/* Show existing groups first */}
                    {patientGroups.length > 0 && (
                      <div className="border-b bg-blue-50">
                        <div className="px-3 py-2 text-xs font-medium text-blue-700 uppercase tracking-wide">
                          Patient Groups
                        </div>
                        <>
                          {patientGroups.map(group => (
                            <div key={group.id} className="flex items-center justify-between">
                              <button
                                className={`flex-1 text-left px-3 py-2 hover:bg-blue-100 border-b last:border-b-0 ${selectedGroup?.id === group.id ? "bg-blue-100 border-blue-300" : ""}`}
                                onClick={() => handleGroupSelect(group)}
                              >
                                <div className="flex items-center gap-2">
                                  <Users className="w-4 h-4 text-blue-600" />
                                  <div className="font-medium text-blue-900">{group.name}</div>
                                </div>
                                <div className="text-xs text-blue-600">
                                  {group.patients.length} patients • Created {formatDate(group.createdAt.toISOString())}
                                </div>
                              </button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  deleteGroup(group.id);
                                }}
                                className="mr-2 h-8 w-8 p-0 text-red-500 hover:text-red-700"
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
                          ))}
                        </>
                      </div>
                    )}
                    
                    {/* Individual patients */}
                    {patientQuery.trim() === '' && patients.length === 0 && patientGroups.length === 0 && (
                      <div className="p-3 text-sm text-gray-500">
                        Start typing to search for patients by name, ID, or date of birth.
                      </div>
                    )}
                    
                    {patientQuery.trim() !== '' && patients.length === 0 && (
                      <div className="p-3 text-sm text-gray-500">
                        No patients found matching "{patientQuery}". Try a different search term.
                      </div>
                    )}
                    
                    {patients.map(p => (
                      <button
                        key={p.id}
                        className={`w-full text-left px-3 py-2 hover:bg-gray-50 border-b last:border-b-0 ${selectedPatient?.id === p.id ? "bg-blue-50 border-blue-200" : ""}`}
                        onClick={() => handlePatientSelect(p)}
                      >
                        <div className="font-medium">{p.fullname}</div>
                        <div className="text-xs text-gray-500 flex items-center gap-2">
                          <span>ID: {p.id}</span>
                          {p.dob && <span>• DOB: {p.dob}</span>}
                        </div>
                      </button>
                    ))}
                  </div>
                  
                  {selectedPatient && (
                    <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
                      <div className="text-sm font-medium text-blue-900">Selected Patient</div>
                      <div className="text-sm text-blue-800">{selectedPatient.fullname}</div>
                      <div className="text-xs text-blue-600">
                        ID: {selectedPatient.id}
                        {selectedPatient.dob && ` • DOB: ${selectedPatient.dob}`}
                      </div>
                    </div>
                  )}
                  
                  {selectedGroup && (
                    <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-md">
                      <div className="flex items-center gap-2 text-sm font-medium text-green-900">
                        <Users className="w-4 h-4" />
                        Selected Group: {selectedGroup.name}
                      </div>
                      <div className="text-sm text-green-800">{selectedGroup.patients.length} patients</div>
                      <div className="text-xs text-green-600 mt-1">
                        {selectedGroup.patients.map(p => p.fullname).join(', ')}
                      </div>
                    </div>
                  )}
                  
                  {!selectedPatient && !selectedGroup && (
                    <div className="flex items-center gap-2 text-amber-600 text-sm mt-2">
                      <ShieldAlert className="w-4 h-4" /> 
                      Please select a patient or create a group to enable medical record search.
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3"><CardTitle className="text-base">Retrieval settings</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label>Results to retrieve</Label>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-auto p-0">
                            <Info className="w-3 h-3 text-gray-400 cursor-help" />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent side="right" className="w-80">
                          <div className="space-y-2">
                            <h4 className="font-medium text-sm">Results to retrieve</h4>
                            <p className="text-xs text-gray-600">
                              Number of most relevant document sections to find before generating an answer. 
                              Higher values provide more context but may be slower. Note that the LLaMA-3.3-70B, which is currently used for generation, has a maximum context length of 128k tokens but studies show that longer contexts can lead to worse results due to attention dilution.
                              {selectedGroup && ` For group queries, this retrieves ${settings.k} results PER PATIENT (total: ${settings.k * selectedGroup.patients.length} results).`}
                            </p>
                          </div>
                        </PopoverContent>
                      </Popover>
                    </div>
                    <Badge variant="secondary">{settings.k}</Badge>
                  </div>
                  <Slider value={[settings.k]} min={3} max={30} step={1} onValueChange={([v]) => setSettings(s => ({ ...s, k: v }))} />
                </div>

                <div>
                  <Label>Search Level</Label>
                  <Select value={settings.searchLevel} onValueChange={v => setSettings(s => ({ ...s, searchLevel: v as any }))}>
                    <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sections">Sections (focused)</SelectItem>
                      <SelectItem value="reports">Full Reports</SelectItem>
                      <SelectItem value="both">Both</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <DateField label="Start" value={settings.startDate} onChange={d => setSettings(s => ({ ...s, startDate: d }))} />
                  <DateField label="End" value={settings.endDate} onChange={d => setSettings(s => ({ ...s, endDate: d }))} />
                </div>

                <Accordion type="single" collapsible>
                  <AccordionItem value="adv">
                    <AccordionTrigger className="text-sm"><Settings className="w-4 h-4 mr-2" /> Advanced</AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-3">
                        <div>
                          <Label>Report types</Label>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {reportTypes.map(dt => (
                              <ToggleChip
                                key={dt}
                                label={dt}
                                active={settings.docTypes.includes(dt)}
                                onToggle={() =>
                                  setSettings(s => ({
                                    ...s,
                                    docTypes: s.docTypes.includes(dt)
                                      ? s.docTypes.filter(x => x !== dt)
                                      : [...s.docTypes, dt],
                                  }))
                                }
                              />
                            ))}
                          </div>
                        </div>

                        {settings.searchLevel !== "reports" && (
                          <div>
                            <Label>Section types</Label>
                            <div className="text-xs text-gray-500 mb-2">Most common sections shown first</div>
                            <div className="flex flex-wrap gap-2 mt-2 max-h-32 overflow-y-auto">
                              {sectionTypes.slice(0, 15).map(st => (
                                <ToggleChip
                                  key={st}
                                  label={st}
                                  active={settings.sectionTypes.includes(st)}
                                  onToggle={() =>
                                    setSettings(s => ({
                                      ...s,
                                      sectionTypes: s.sectionTypes.includes(st)
                                        ? s.sectionTypes.filter(x => x !== st)
                                        : [...s.sectionTypes, st],
                                    }))
                                  }
                                />
                              ))}
                            </div>
                          </div>
                        )}

                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <Label>Custom Prompt</Label>
                            <Switch
                              checked={settings.useCustomPrompt}
                              onCheckedChange={v => setSettings(s => ({ ...s, useCustomPrompt: v }))}
                            />
                          </div>
                          
                          {settings.useCustomPrompt && (
                            <div className="space-y-2">
                              <Select
                                value=""
                                onValueChange={(presetKey) => {
                                  const preset = PROMPT_PRESETS[presetKey as keyof typeof PROMPT_PRESETS];
                                  if (preset) {
                                    setSettings(s => ({ ...s, customPrompt: preset.prompt }));
                                  }
                                }}
                              >
                                <SelectTrigger>
                                  <SelectValue placeholder="Choose preset..." />
                                </SelectTrigger>
                                <SelectContent>
                                  {Object.entries(PROMPT_PRESETS).map(([key, preset]) => (
                                    <SelectItem key={key} value={key}>
                                      <div>
                                        <div className="font-medium">{preset.name}</div>
                                        <div className="text-xs text-gray-500">{preset.description}</div>
                                      </div>
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              
                              <textarea
                                value={settings.customPrompt}
                                onChange={e => setSettings(s => ({ ...s, customPrompt: e.target.value }))}
                                placeholder="Enter custom prompt..."
                                rows={6}
                                className="w-full p-2 text-xs border rounded resize-none"
                              />
                            </div>
                          )}
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <Label>Score Fallback</Label>
                            <Switch
                              checked={settings.enableScoreFallback}
                              onCheckedChange={v => setSettings(s => ({ ...s, enableScoreFallback: v }))}
                            />
                          </div>
                          
                          {settings.enableScoreFallback && (
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Label className="text-xs">Min Score Threshold</Label>
                                <Badge variant="secondary">{settings.minScoreThreshold}</Badge>
                              </div>
                              <Slider
                                value={[settings.minScoreThreshold]}
                                min={0.1}
                                max={1.0}
                                step={0.1}
                                onValueChange={([v]) => setSettings(s => ({ ...s, minScoreThreshold: v }))}
                              />
                            </div>
                          )}
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CardContent>
            </Card>

            {/* Simplified Logging Configuration Card */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Session Logging</CardTitle>
                <CardDescription>Configure automatic logging of queries and results</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Enable Logging</Label>
                  <Switch
                    checked={loggingConfig.enabled}
                    onCheckedChange={v => setLoggingConfig(s => ({ ...s, enabled: v }))}
                  />
                </div>

                {loggingConfig.enabled && (
                  <>
                    <div>
                      <Label>Log File Path</Label>
                      <Input
                        placeholder="/path/to/logfile"
                        value={loggingConfig.filepath}
                        onChange={e => setLoggingConfig(s => ({ ...s, filepath: e.target.value }))}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label>Format</Label>
                      <Select value={loggingConfig.format} onValueChange={v => setLoggingConfig(s => ({ ...s, format: v as 'json' | 'csv' }))}>
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="json">JSON</SelectItem>
                          <SelectItem value="csv">CSV</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Include Full Content</Label>
                      <Switch
                        checked={loggingConfig.includeFullContent}
                        onCheckedChange={v => setLoggingConfig(s => ({ ...s, includeFullContent: v }))}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Auto-log After Each Query</Label>
                      <Switch
                        checked={loggingConfig.autoLog ?? true} // Use true as fallback for existing configs
                        onCheckedChange={v => setLoggingConfig(s => ({ ...s, autoLog: v }))}
                      />
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Main content */}
          <div className="lg:col-span-3 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Ask a question</CardTitle>
                <CardDescription>
                  {selectedGroup 
                    ? `Ask questions about the ${selectedGroup.patients.length} patients in "${selectedGroup.name}"`
                    : selectedPatient 
                      ? `Ask questions about ${selectedPatient.fullname}'s medical records`
                      : "Select a patient or group to start asking questions"
                  }
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <textarea
                  placeholder="What do you want to know?"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  rows={3}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
                />
                
                <div className="flex gap-2">
                  <Button onClick={run} disabled={!canRun || isRunning} className="flex-1">
                    {isRunning ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Search className="w-4 h-4 mr-2" />}
                    {isRunning ? "Searching..." : "Search"}
                  </Button>
                  <Button variant="outline" onClick={clearAll}>
                    <TimerReset className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Results */}
            {(hits.length > 0 || answer) && (
              <Tabs defaultValue="answer" className="w-full">
                <TabsList>
                  <TabsTrigger value="answer">Answer</TabsTrigger>
                  <TabsTrigger value="sources">Sources ({hits.length})</TabsTrigger>
                </TabsList>
                
                <TabsContent value="answer">
                  <Card>
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base">Generated Answer</CardTitle>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm" onClick={copyWithCitations}>
                            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                          </Button>
                          {/* Add logging button */}
                          {canLog && (
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={logCurrentSession}
                              disabled={isLogging}
                            >
                              {isLogging ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                            </Button>
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {answer ? (
                        <div className="space-y-3">
                          <div className="prose max-w-none">
                            <p className="whitespace-pre-wrap">{answer.text}</p>
                          </div>
                          
                          {answer.warnings && answer.warnings.length > 0 && (
                            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-md">
                              <div className="flex items-center gap-2 text-amber-800 text-sm font-medium">
                                <ShieldAlert className="w-4 h-4" />
                                Warnings
                              </div>
                              <ul className="mt-1 text-sm text-amber-700">
                                {answer.warnings.map((w, i) => (
                                  <li key={i}>• {w}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          No answer generated yet
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="sources">
                  <div className="space-y-3">
                    {hits.map((hit, i) => (
                      <Card key={hit.id} className={`${hit.pinned ? "ring-2 ring-blue-500" : ""}`}>
                        <CardContent className="pt-4">
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-2">
                                {hit.patientName && (
                                  <Badge variant="outline" className="text-xs">
                                    <User className="w-3 h-3 mr-1" />
                                    {hit.patientName}
                                  </Badge>
                                )}
                                <Badge variant="secondary">{hit.docType}</Badge>
                                {hit.section && <Badge variant="outline">{hit.section}</Badge>}
                                <Badge variant="outline">{formatDate(hit.date)}</Badge>
                                {hit.score && (
                                  <Badge variant="outline">
                                    Score: {hit.score.toFixed(2)}
                                  </Badge>
                                )}
                              </div>
                              
                              <div className="text-sm text-gray-700 mb-2">
                                {hit.snippet.substring(0, 300)}
                                {hit.snippet.length > 300 && "..."}
                              </div>
                              
                              <div className="flex items-center gap-2 text-xs text-gray-500">
                                {hit.filename && <span>📄 {hit.filename}</span>}
                                {hit.wordCount && <span>• {hit.wordCount} words</span>}
                              </div>
                            </div>
                            
                            <div className="flex gap-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => togglePin(hit.id)}
                                className={hit.pinned ? "text-blue-600" : ""}
                              >
                                <Plus className="w-4 h-4" />
                              </Button>
                              
                              {hit.reportId && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => openReport(hit)}
                                >
                                  <ExternalLink className="w-4 h-4" />
                                </Button>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </div>

          {/* Right column: Log Preview */}
          {loggingConfig.enabled && loggingConfig.filepath && (
            <div className="lg:col-span-1">
              <Card className="sticky top-24">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">Log Preview</CardTitle>
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={fetchLogContent}
                        disabled={isLoadingLogs}
                        title="Refresh log content"
                      >
                        {isLoadingLogs ? <Loader2 className="w-3 h-3 animate-spin" /> : "Refresh"}
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={copyLogContent}
                        disabled={!logContent || logContent.trim() === ''}
                        title={`Copy ${logContent?.length || 0} characters to clipboard`}
                      >
                        {logCopied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                      </Button>
                    </div>
                  </div>
                  <CardDescription className="text-xs">
                    Live feed from {loggingConfig.filepath}
                    {logContent && (
                      <span className="ml-2 text-gray-400">
                        ({logContent.length} chars)
                      </span>
                    )}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96 w-full">
                    <pre className="text-xs whitespace-pre-wrap break-words font-mono">
                      {logContent || 'No log content available'}
                    </pre>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          )}
        </main>

        {/* Group Builder Modal - Fixed */}
        {showGroupBuilder && (
                   <div className="fixed inset-0 z-50 bg-black/80" onClick={() => setShowGroupBuilder(false)}>
            <div className="fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 sm:rounded-lg" onClick={e => e.stopPropagation()}>
              <div className="flex flex-col space-y-1.5 text-center sm:text-left">
                <h2 className="text-lg font-semibold leading-none tracking-tight">Build Patient Group</h2>
                <p className="text-sm text-muted-foreground">Create a group of patients for comparative analysis</p>
              </div>
              
              <div className="space-y-4">
                <div>
                  <Label>Group Name</Label>
                  <Input 
                    placeholder="Enter group name..." 
                    value={groupName}
                    onChange={e => setGroupName(e.target.value)}
                  />
                </div>
                
                <div>
                  <Label>Selected Patients ({selectedPatientIds.length} of {patients.length})</Label>
                  <div className="mt-2 max-h-48 border rounded-md bg-white overflow-auto">
                    {patients.map(patient => (
                      <div key={patient.id} className="flex items-center gap-2 p-2 border-b last:border-b-0">
                        <input 
                          type="checkbox" 
                          className="rounded"
                          checked={selectedPatientIds.includes(patient.id)}
                          onChange={() => togglePatientSelection(patient.id)}
                        />
                        <div className="flex-1">
                          <div className="font-medium">{patient.fullname}</div>
                          <div className="text-xs text-gray-500">ID: {patient.id}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="flex gap-2">
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setShowGroupBuilder(false);
                    setGroupName("");
                    setSelectedPatientIds([]);
                  }} 
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button 
                  onClick={createGroup}
                  disabled={!groupName.trim() || selectedPatientIds.length === 0}
                  className="flex-1"
                >
                  Create Group
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Report Viewer Modal */}
        {selectedReport && (
          <div className="fixed inset-0 z-50 bg-black/80" onClick={() => setSelectedReport(null)}>
            <div className="fixed left-[50%] top-[50%] z-50 flex flex-col w-full max-w-4xl h-[90vh] translate-x-[-50%] translate-y-[-50%] border bg-background shadow-lg duration-200 sm:rounded-lg" onClick={e => e.stopPropagation()}>
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b flex-shrink-0">
                <div>
                  <h2 className="text-lg font-semibold leading-none tracking-tight">
                    {selectedReport.filename || "Report"}
                  </h2>
                  <div className="text-sm font-medium text-gray-500">
                    {selectedReport.docType} • {formatDate(selectedReport.date)}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedReport(null)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              {/* Content with ScrollArea */}
              <div className="flex-1 overflow-hidden">
                {selectedReport.highlightSection && (
                  <div className="px-6 py-2 bg-blue-50 border-b text-sm text-blue-700 flex-shrink-0">
                    Highlighted section from {selectedReport.patientId}: {selectedReport.highlightSection}
                  </div>
                )}
                
                <ScrollArea className="h-full">
                  <div className="p-6">
                    <div
                      className="prose max-w-none text-sm"
                      dangerouslySetInnerHTML={{ __html: selectedReport.highlightedContent || selectedReport.content }}
                    />
                  </div>
                </ScrollArea>
              </div>
              
              {/* Footer */}
              <div className="p-4 border-t bg-gray-50 flex items-center justify-between text-xs text-gray-500 flex-shrink-0">
                <span>Patient ID: {selectedReport.patientId}</span>
                <span>Report ID: {selectedReport.reportId}</span>
                {selectedReport.highlightSection && (
                  <span className="text-blue-600">Section: {selectedReport.highlightSection}</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}
