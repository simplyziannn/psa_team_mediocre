import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, AlertTriangle, ArrowUpCircle } from "lucide-react";

export interface IncidentAnalysis {
  incident: string;
  resolution: string;
  escalationNeeded: boolean;
  escalationMessage?: string;
  severity?: "low" | "medium" | "high" | "critical";
  timestamp: Date;
}

interface IncidentResultProps {
  analysis: IncidentAnalysis;
}

export const IncidentResult = ({ analysis }: IncidentResultProps) => {
  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case "critical":
        return "destructive";
      case "high":
        return "warning";
      case "medium":
        return "secondary";
      case "low":
      default:
        return "muted";
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto space-y-4 animate-fade-in">
      {/* Incident Description */}
      <Card className="p-6 bg-card border-border">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-muted-foreground mt-1 flex-shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <h3 className="text-sm font-medium text-muted-foreground">Incident Report</h3>
              {analysis.severity && (
                <Badge variant={getSeverityColor(analysis.severity) as any} className="capitalize">
                  {analysis.severity}
                </Badge>
              )}
            </div>
            <p className="text-foreground leading-relaxed">{analysis.incident}</p>
            <p className="text-xs text-muted-foreground">
              {analysis.timestamp.toLocaleString()}
            </p>
          </div>
        </div>
      </Card>

      {/* Resolution */}
      <Card className="p-6 bg-card border-border">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="h-5 w-5 text-primary mt-1 flex-shrink-0" />
          <div className="flex-1 space-y-2">
            <h3 className="text-sm font-medium text-primary">Recommended Resolution</h3>
            <p className="text-foreground leading-relaxed whitespace-pre-wrap">
              {analysis.resolution}
            </p>
          </div>
        </div>
      </Card>

      {/* Escalation */}
      {analysis.escalationNeeded && analysis.escalationMessage && (
        <Card className="p-6 bg-warning/10 border-warning/30">
          <div className="flex items-start gap-3">
            <ArrowUpCircle className="h-5 w-5 text-warning mt-1 flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <h3 className="text-sm font-medium text-warning">Escalation Required</h3>
              <p className="text-foreground leading-relaxed whitespace-pre-wrap">
                {analysis.escalationMessage}
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
