import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send, Loader2 } from "lucide-react";

interface IncidentInputProps {
  onSubmit: (description: string) => void;
  isLoading: boolean;
}

export const IncidentInput = ({ onSubmit, isLoading }: IncidentInputProps) => {
  const [description, setDescription] = useState("");

  const handleSubmit = () => {
    if (description.trim()) {
      onSubmit(description);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto space-y-4">
      <div className="relative">
        <Textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe the incident..."
          className="min-h-[120px] resize-none bg-secondary border-border focus:border-primary transition-colors pr-14"
          disabled={isLoading}
        />
        <Button
          onClick={handleSubmit}
          disabled={!description.trim() || isLoading}
          size="icon"
          className="absolute bottom-3 right-3 rounded-full bg-primary hover:bg-primary/90 transition-all"
        >
          {isLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Send className="h-5 w-5" />
          )}
        </Button>
      </div>
      <p className="text-sm text-muted-foreground text-center">
        Press <kbd className="px-2 py-1 text-xs bg-muted rounded">Cmd/Ctrl</kbd> +{" "}
        <kbd className="px-2 py-1 text-xs bg-muted rounded">Enter</kbd> to submit
      </p>
    </div>
  );
};
