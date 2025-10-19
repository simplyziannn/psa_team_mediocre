import { useState } from "react";
import { IncidentInput } from "@/components/IncidentInput";
import { IncidentResult, IncidentAnalysis } from "@/components/IncidentResult";
import { Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<IncidentAnalysis | null>(null);
  const [isDark, setIsDark] = useState(true);

  const handleSubmit = async (description: string) => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: description }),
      });
      const bodyAsText = await res.text();
      let data: any = null;
      try {
        data = bodyAsText ? JSON.parse(bodyAsText) : null;
      } catch (err) {
        throw new Error(`Invalid JSON from server: ${bodyAsText || String(err)}`);
      }

      // API returns { ok: true, result: {...} } or { ok: false, error: '...' }
      if (!data || data.ok === false) {
        const err = data?.error || (data?.result && JSON.stringify(data.result)) || `Server error: ${res.status}`;
        setAnalysis({
          incident: description,
          resolution: `❌ Failed to process:\n\n${String(err)}`,
          escalationNeeded: false,
          severity: 'high',
          timestamp: new Date(),
        });
        return;
      }

      const result = data.result || {};
      
      // Check for pipeline errors
      if (result.error) {
        setAnalysis({
          incident: description,
          resolution: `❌ Pipeline Error:\n\n${result.error}\n\nStage: ${result.stage || 'Unknown'}`,
          escalationNeeded: false,
          severity: 'high',
          timestamp: new Date(),
        });
        return;
      }

      const decision = result.decision || {};
      const compiled = result.compiled || {};
      const dbResult = result.db_result || {};
      const pdfOutput = result.pdf_output || [];
      const problemDraft = result.problem_draft || {};

      // Build resolution text with proper formatting
      let resolutionParts: string[] = [];
      
      // Show extracted incident info first
      if (problemDraft.incident_type) {
        resolutionParts.push(`📋 Incident Type: ${problemDraft.incident_type}`);
        resolutionParts.push(`Confidence: ${(problemDraft.confidence * 100).toFixed(0)}%`);
        resolutionParts.push('');
      }
      
      // If we have a decision, show it nicely
      if (decision?.json) {
        const dec = decision.json;
        resolutionParts.push(`🎯 Module: ${dec.module || 'N/A'}`);
        resolutionParts.push('');
        if (dec.summary) {
          resolutionParts.push(`📝 Summary:`);
          resolutionParts.push(dec.summary);
          resolutionParts.push('');
        }
        if (dec.root_cause) {
          resolutionParts.push(`🔍 Root Cause:`);
          resolutionParts.push(dec.root_cause);
          resolutionParts.push('');
        }
        if (dec.resolution_steps && Array.isArray(dec.resolution_steps)) {
          resolutionParts.push('🛠️ Resolution Steps:');
          dec.resolution_steps.forEach((step: string, idx: number) => {
            resolutionParts.push(`${idx + 1}. ${step}`);
          });
          resolutionParts.push('');
        }
      } else if (decision?.text) {
        resolutionParts.push('📄 Analysis:');
        resolutionParts.push(decision.text);
        resolutionParts.push('');
      } else if (decision?.error) {
        resolutionParts.push('⚠️ Decision Engine Error:');
        resolutionParts.push(decision.error);
        resolutionParts.push('');
      }
      
      // Show compiled data if no decision yet
      if (!decision?.json && !decision?.text && compiled && Object.keys(compiled).length > 0) {
        if (compiled.alert_result?.owner?.sections) {
          const overview = compiled.alert_result.owner.sections.find((s: any) => s.title === 'Overview');
          const resolution = compiled.alert_result.owner.sections.find((s: any) => s.title === 'Resolution');
          
          if (overview?.lines) {
            resolutionParts.push('📖 Overview:');
            overview.lines.forEach((line: string) => resolutionParts.push(line));
            resolutionParts.push('');
          }
          
          if (resolution?.lines) {
            resolutionParts.push('🛠️ Resolution Steps:');
            resolution.lines.forEach((line: string, idx: number) => {
              resolutionParts.push(`${idx + 1}. ${line}`);
            });
            resolutionParts.push('');
          }
        }
      }
      
      // Show database matches if available
      if (dbResult?.matches && Array.isArray(dbResult.matches) && dbResult.matches.length > 0) {
        resolutionParts.push('📦 Database Matches:');
        dbResult.matches.forEach((match: any) => {
          resolutionParts.push(`\n  Container: ${match.cntr_no} (${match.size_type})`);
          resolutionParts.push(`  Status: ${match.status}`);
          resolutionParts.push(`  Route: ${match.origin_port} → ${match.destination_port}`);
          if (match.vessel_id) {
            resolutionParts.push(`  Vessel ID: ${match.vessel_id}`);
          }
          if (match.eta_ts) {
            resolutionParts.push(`  ETA: ${match.eta_ts}`);
          }
        });
        resolutionParts.push('');
      }

      const resolutionText = resolutionParts.length > 0 
        ? resolutionParts.join('\n') 
        : '⚠️ No detailed analysis available. Raw data:\n' + JSON.stringify(result, null, 2);

      // Build escalation message
      let escalationParts: string[] = [];
      const esc = decision?.json?.escalation || compiled?.escalation;
      if (esc) {
        if (esc.target) {
          escalationParts.push(`🎯 Target: ${esc.target}`);
          escalationParts.push('');
        }
        if (esc.contacts && Array.isArray(esc.contacts) && esc.contacts.length > 0) {
          escalationParts.push('👥 Contacts:');
          esc.contacts.forEach((c: any) => {
            escalationParts.push(`  • ${c.name} <${c.email}>`);
          });
          escalationParts.push('');
        }
        if (esc.steps && Array.isArray(esc.steps) && esc.steps.length > 0) {
          escalationParts.push('📝 Escalation Steps:');
          esc.steps.forEach((step: string, idx: number) => {
            escalationParts.push(`  ${idx + 1}. ${step}`);
          });
        }
      } else if (pdfOutput.length > 0) {
        // Fallback to PDF output
        escalationParts.push('👥 Escalation Contacts:');
        escalationParts.push('');
        pdfOutput.forEach((contact: any) => {
          escalationParts.push(`Module: ${contact.module || 'N/A'}`);
          escalationParts.push(`Contact: ${contact.manager_name || 'N/A'}`);
          if (contact.emails && Array.isArray(contact.emails)) {
            escalationParts.push(`Email: ${contact.emails.join(', ')}`);
          }
          if (contact.role) {
            escalationParts.push(`Role: ${contact.role}`);
          }
          if (contact.escalation_steps && Array.isArray(contact.escalation_steps)) {
            escalationParts.push('Steps:');
            contact.escalation_steps.forEach((step: string) => {
              escalationParts.push(`  • ${step}`);
            });
          }
          escalationParts.push('');
        });
      }

      const analysis: IncidentAnalysis = {
        incident: description,
        resolution: resolutionText,
        escalationNeeded: escalationParts.length > 0,
        escalationMessage: escalationParts.join('\n') || undefined,
        severity: problemDraft.confidence > 0.7 ? 'high' : problemDraft.confidence > 0.4 ? 'medium' : 'low',
        timestamp: new Date(),
      };

      setAnalysis(analysis);
    } catch (e) {
      setAnalysis({
        incident: description,
        resolution: `Failed to process: ${String(e)}`,
        escalationNeeded: false,
        severity: 'medium',
        timestamp: new Date(),
      });
    } finally {
      setIsLoading(false);
    }
  };

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle("dark");
  };

  return (
    <div className={`min-h-screen bg-background transition-colors ${isDark ? "dark" : ""}`}>
      {/* Background gradient effect */}
      <div 
        className="fixed inset-0 pointer-events-none"
        style={{ background: "var(--gradient-radial)" }}
      />
      
      {/* Header */}
      <header className="relative border-b border-border">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-foreground">Incident Processor</h1>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="rounded-full"
          >
            {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      {/* Main content */}
      <main className="relative container mx-auto px-4 py-12">
        {!analysis ? (
          <div className="flex flex-col items-center justify-center min-h-[calc(100vh-12rem)]">
            <div className="text-center space-y-6 mb-12">
              <h2 className="text-4xl md:text-5xl font-bold text-foreground">
                How can I help?
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl">
                Describe any incident to receive AI-powered analysis
              </p>
            </div>
            <IncidentInput onSubmit={handleSubmit} isLoading={isLoading} />
          </div>
        ) : (
          <div className="space-y-6">
            <Button
              variant="ghost"
              onClick={() => setAnalysis(null)}
              className="mb-4"
            >
              ← New Incident
            </Button>
            <IncidentResult analysis={analysis} />
          </div>
        )}
      </main>
    </div>
  );
};

export default Index;
