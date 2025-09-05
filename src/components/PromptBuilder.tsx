import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

// Types
type Suggestion = string;
type TokenRole = "descriptor" | "quality" | "mood" | "action";

type Token = { 
  id: string; 
  text: string; 
  role: TokenRole; 
  alts: string[] 
};

type PromptDoc = { 
  raw: string; 
  tokens: Token[] 
};

type LLMApi = {
  complete(text: string): Promise<{ suggestions: Suggestion[] }>;
  enhance(selection: string): Promise<{ prompt: string }>;
  alternatives(prompt: string): Promise<{ tokens: Token[] }>;
};

// Utils
let idCounter = 0;
const uid = () => `token-${Date.now()}-${++idCounter}`;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Token styling configuration
const ROLE_COLORS: Record<TokenRole, string> = {
  descriptor: "bg-token-descriptor-bg text-token-descriptor border-token-descriptor-border underline decoration-token-descriptor/70",
  quality: "bg-token-quality-bg text-token-quality border-token-quality-border underline decoration-token-quality/70", 
  mood: "bg-token-mood-bg text-token-mood border-token-mood-border underline decoration-token-mood/70",
  action: "bg-token-action-bg text-token-action border-token-action-border underline decoration-token-action/70",
};

// Keyword-based role inference
const KEYWORD_ROLE: Array<{ pattern: RegExp; role: TokenRole }> = [
  { 
    pattern: /(high\s+quality|8k|ultra\s+detail|studio\s+lighting|volumetric|rim\s+light|softbox|hard\s+light|backlit|bokeh)/gi, 
    role: "quality" 
  },
  { 
    pattern: /(chiaroscuro|cinematic|cool\s+tones|warm\s+tones|neon|pastel|monochrome|earthy|desaturated|moody|vibrant)/gi, 
    role: "mood" 
  },
  { 
    pattern: /(carved|woven|forged|grown|assembled|whittled|sculpted|panning|tilt|dolly|zoom|tracking|orbiting)/gi, 
    role: "action" 
  },
  { 
    pattern: /(dress|gown|armor|cat|city|forest|portrait|mountain|desert|robot|mask|character|creature|temple|street)/gi, 
    role: "descriptor" 
  },
];

function inferRole(text: string): TokenRole {
  for (const { pattern, role } of KEYWORD_ROLE) {
    if (pattern.test(text)) return role;
  }
  
  // Fallback heuristics
  if (/\b(ly|ic|ous|ful|ish|ive)\b/i.test(text) || /light|tone|quality|detail/i.test(text)) 
    return "quality";
  if (/\b(run|fly|carved|forged|grown|shoot|tilt|pan|zoom|orbit|track|hold)\b/i.test(text)) 
    return "action";
  if (/\b(moody|warm|cool|neon|pastel|earthy|grim|bright|soft|harsh)\b/i.test(text)) 
    return "mood";
  
  return "descriptor";
}

// Mock LLM for demo purposes
const MockLLM: LLMApi = {
  async complete(text) {
    await sleep(200);
    const base = text.trim();
    if (!base) return { suggestions: [] };
    
    const ideas = [
      `${base} in cinematic lighting`,
      `${base} at golden hour`,
      `${base} with dramatic shadows`,
    ];
    
    return { suggestions: ideas.slice(0, 3) };
  },
  
  async enhance(selection) {
    await sleep(300);
    return { 
      prompt: `${selection} — high quality, ultra detailed, studio lighting, cinematic, cool tones, chiaroscuro, softbox lighting.`
    };
  },
  
  async alternatives(prompt) {
    await sleep(200);
    
    // Extract key terms and create tokens
    const candidates = [
      "high quality", "ultra detailed", "studio lighting", "cinematic", 
      "cool tones", "chiaroscuro", "softbox lighting", "dramatic", "portrait"
    ].filter(term => prompt.toLowerCase().includes(term.toLowerCase()));

    const altMap: Record<string, string[]> = {
      "high quality": ["premium", "photoreal", "film-grade", "clean", "polished", "pristine"],
      "ultra detailed": ["intricate", "highly detailed", "fine detail", "micro-detail", "ornate", "textured"],
      "studio lighting": ["natural light", "softbox", "rim light", "backlit", "hard light", "volumetric"],
      "cinematic": ["filmic", "dramatic", "epic", "documentary", "stylized", "handheld"],
      "cool tones": ["warm tones", "pastel", "neon", "monochrome", "earthy", "muted"],
      "chiaroscuro": ["high contrast", "low key", "Rembrandt light", "split lighting", "soft contrast", "flat light"],
      "softbox lighting": ["umbrella light", "beauty dish", "diffused light", "bounce light", "window light", "ring light"],
      "dramatic": ["subtle", "intense", "moody", "striking", "bold", "understated"],
      "portrait": ["landscape", "close-up", "wide shot", "macro", "aerial", "street scene"],
    };

    const tokens: Token[] = candidates.map(text => ({
      id: uid(),
      text,
      role: inferRole(text),
      alts: altMap[text] || ["variant A", "variant B", "variant C", "variant D", "variant E", "variant F"]
    }));

    // Ensure we have at least 8 tokens by adding synthetic ones
    while (tokens.length < 8) {
      const syntheticTerms = ["dramatic", "vibrant", "elegant", "mysterious", "ethereal", "bold"];
      const term = syntheticTerms[tokens.length % syntheticTerms.length];
      tokens.push({
        id: uid(),
        text: term,
        role: inferRole(term),
        alts: ["striking", "subtle", "intense", "gentle", "powerful", "delicate"]
      });
    }

    return { tokens: tokens.slice(0, 12) };
  },
};

// Prompt segmentation
type Segment = 
  | { kind: "text"; value: string }
  | { kind: "token"; tokenId: string };

function segmentPrompt(raw: string, tokens: Token[]): Segment[] {
  if (!raw) return [];
  if (tokens.length === 0) return [{ kind: "text", value: raw }];

  // Sort tokens by length (longest first) to handle overlaps
  const sortedTokens = [...tokens].sort((a, b) => b.text.length - a.text.length);
  
  let working = raw;
  const placements: Array<{ start: number; end: number; tokenId: string }> = [];

  for (const token of sortedTokens) {
    const pattern = new RegExp(`\\b${escapeRegExp(token.text)}\\b`, "i");
    const match = pattern.exec(working);
    
    if (match) {
      const start = match.index;
      const end = start + match[0].length;
      
      placements.push({ start, end, tokenId: token.id });
      
      // Replace matched text with placeholders to avoid overlaps
      working = working.slice(0, start) + "█".repeat(end - start) + working.slice(end);
    }
  }

  // Sort placements by position
  placements.sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let currentIndex = 0;

  for (const placement of placements) {
    // Add text before token
    if (placement.start > currentIndex) {
      segments.push({ 
        kind: "text", 
        value: raw.slice(currentIndex, placement.start) 
      });
    }

    // Add token segment
    segments.push({ 
      kind: "token", 
      tokenId: placement.tokenId 
    });

    currentIndex = placement.end;
  }

  // Add remaining text
  if (currentIndex < raw.length) {
    segments.push({ 
      kind: "text", 
      value: raw.slice(currentIndex) 
    });
  }

  return segments.length ? segments : [{ kind: "text", value: raw }];
}

function escapeRegExp(string: string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function buildRawFromSegments(segments: Segment[], tokenMap: Map<string, Token>): string {
  return segments
    .map(segment => 
      segment.kind === "text" 
        ? segment.value 
        : tokenMap.get(segment.tokenId)?.text ?? ""
    )
    .join("");
}

// UI Components
interface HeaderProps {
  live: boolean;
  setLive: (value: boolean) => void;
}

function Header({ live, setLive }: HeaderProps) {
  return (
    <header className="flex items-center justify-between py-6">
      <div>
        <h1 className="text-3xl font-bold bg-gradient-hero bg-clip-text text-transparent">
          Prompt Builder
        </h1>
        <p className="text-muted-foreground mt-1">
          Enhance your prompts with AI-powered suggestions
        </p>
      </div>
      
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          <input
            type="checkbox"
            checked={live}
            onChange={(e) => setLive(e.target.checked)}
            className="rounded border-gray-300"
          />
          Live suggestions
        </label>
      </div>
    </header>
  );
}

interface ChipsProps {
  items: Suggestion[];
  onPick: (suggestion: Suggestion) => void;
  loading: boolean;
}

function Chips({ items, onPick, loading }: ChipsProps) {
  if (loading) {
    return (
      <div className="flex gap-3 flex-wrap animate-pulse">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-10 w-48 rounded-full bg-muted" />
        ))}
      </div>
    );
  }

  return (
    <div className="flex gap-3 flex-wrap">
      {items.map((suggestion, i) => (
        <Button
          key={i}
          variant="outline"
          onClick={() => onPick(suggestion)}
          className="rounded-full hover:shadow-soft transition-all duration-200 hover:-translate-y-0.5"
        >
          {suggestion}
        </Button>
      ))}
    </div>
  );
}

interface TokenPillProps {
  token: Token;
  onClick: () => void;
}

function TokenPill({ token, onClick }: TokenPillProps) {
  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center px-2 py-1 -mx-0.5 rounded-md border text-sm font-medium
        transition-all duration-200 hover:shadow-soft hover:-translate-y-0.5
        ${ROLE_COLORS[token.role]}
      `}
    >
      {token.text}
    </button>
  );
}

interface DropdownProps {
  token: Token;
  onSelect: (value: string) => void;
  onClose: () => void;
}

function Dropdown({ token, onSelect, onClose }: DropdownProps) {
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    document.addEventListener("mousedown", handleClickOutside);

    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [onClose]);

  return (
    <div
      ref={wrapRef}
      className="absolute z-50 mt-2 min-w-[240px] max-w-[320px] rounded-xl border bg-popover shadow-elegant p-4 animate-in slide-in-from-top-2"
    >
      <div className="text-xs uppercase tracking-wide text-muted-foreground mb-3 font-semibold">
        {token.role}
      </div>
      
      <div className="flex flex-wrap gap-2">
        {[token.text, ...token.alts].slice(0, 7).map((alt, i) => (
          <Button
            key={i}
            variant={i === 0 ? "secondary" : "outline"}
            size="sm"
            onClick={() => onSelect(alt)}
            className="text-xs transition-all duration-200 hover:scale-105"
          >
            {alt}
          </Button>
        ))}
      </div>
    </div>
  );
}

// Main Component
export function PromptBuilder() {
  const [input, setInput] = useState("");
  const [loadingA, setLoadingA] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [live, setLive] = useState(true);

  const [selection, setSelection] = useState<string | null>(null);
  const [loadingB, setLoadingB] = useState(false);
  const [promptDoc, setPromptDoc] = useState<PromptDoc | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [openTokenId, setOpenTokenId] = useState<string | null>(null);

  const tokenMap = useMemo(() => 
    new Map((promptDoc?.tokens ?? []).map(token => [token.id, token])),
    [promptDoc]
  );

  const completeReqId = useRef(0);

  // Live suggestions
  useEffect(() => {
    if (!live) return;
    
    const query = input.trim();
    if (!query) {
      setSuggestions([]);
      setLoadingA(false);
      return;
    }

    setLoadingA(true);
    const reqId = ++completeReqId.current;
    
    const timer = setTimeout(async () => {
      try {
        const result = await MockLLM.complete(query);
        if (reqId === completeReqId.current) {
          setSuggestions(result.suggestions || []);
          setLoadingA(false);
        }
      } catch (error) {
        if (reqId === completeReqId.current) {
          setLoadingA(false);
        }
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [input, live]);

  const handleSuggest = async () => {
    setOpenTokenId(null);
    setSelection(null);
    setPromptDoc(null);
    setSegments([]);

    if (!input.trim()) return;

    setLoadingA(true);
    try {
      const result = await MockLLM.complete(input);
      setSuggestions(result.suggestions || []);
    } finally {
      setLoadingA(false);
    }
  };

  const pickSuggestion = async (suggestion: string) => {
    setSelection(suggestion);
    setLoadingB(true);

    try {
      const enhanced = await MockLLM.enhance(suggestion);
      const altResult = await MockLLM.alternatives(enhanced.prompt);
      
      const segments = segmentPrompt(enhanced.prompt, altResult.tokens);
      
      setPromptDoc({ raw: enhanced.prompt, tokens: altResult.tokens });
      setSegments(segments);
    } finally {
      setLoadingB(false);
    }
  };

  const replaceToken = (tokenId: string, newText: string) => {
    if (!promptDoc) return;

    const updatedTokens = promptDoc.tokens.map(token =>
      token.id === tokenId ? { ...token, text: newText } : token
    );

    const updatedTokenMap = new Map(
      updatedTokens.map(token => [token.id, token])
    );
    
    const updatedRaw = buildRawFromSegments(segments, updatedTokenMap);

    setPromptDoc({ raw: updatedRaw, tokens: updatedTokens });
    setOpenTokenId(null);
  };

  const copyToClipboard = async () => {
    if (!promptDoc?.raw) return;
    
    try {
      await navigator.clipboard.writeText(promptDoc.raw);
      // You could add a toast notification here
      alert("Copied to clipboard!");
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const reset = () => {
    setInput("");
    setSuggestions([]);
    setSelection(null);
    setPromptDoc(null);
    setSegments([]);
    setOpenTokenId(null);
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <Header live={live} setLive={setLive} />

      {/* Input Section */}
      <div className="space-y-6">
        <div className="flex gap-3">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSuggest();
            }}
            placeholder="Type a word or sentence to get started..."
            className="flex-1 h-12 rounded-xl text-lg px-4 shadow-soft"
          />
          <Button 
            onClick={handleSuggest}
            size="lg"
            className="bg-gradient-primary hover:shadow-glow transition-all duration-300 px-8 rounded-xl"
          >
            Suggest
          </Button>
        </div>

        {/* Live preview hint */}
        <div className="min-h-[24px] text-sm text-muted-foreground">
          {loadingA ? (
            <span className="animate-pulse">Finding suggestions...</span>
          ) : suggestions[0] ? (
            <span className="opacity-70">→ {suggestions[0]}</span>
          ) : null}
        </div>

        {/* Suggestion chips */}
        <Chips 
          items={suggestions} 
          loading={loadingA} 
          onPick={pickSuggestion} 
        />

        {/* Prompt Canvas */}
        <div className="rounded-2xl border bg-gradient-subtle shadow-soft p-6 min-h-[200px] relative">
          {!selection && !loadingB && (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              Pick a suggestion to build your enhanced prompt...
            </div>
          )}
          
          {loadingB && (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              <span className="animate-pulse">Crafting enhanced prompt...</span>
            </div>
          )}
          
          {promptDoc && (
            <div className="leading-8 text-lg">
              {segments.map((segment, i) => {
                if (segment.kind === "text") {
                  return <span key={i}>{segment.value}</span>;
                }

                const token = tokenMap.get(segment.tokenId);
                if (!token) return <span key={i}></span>;

                return (
                  <span key={i} className="relative inline-block">
                    <TokenPill
                      token={token}
                      onClick={() => setOpenTokenId(
                        token.id === openTokenId ? null : token.id
                      )}
                    />
                    {openTokenId === token.id && (
                      <div className="absolute left-0 top-full z-50">
                        <Dropdown
                          token={token}
                          onSelect={(value) => replaceToken(token.id, value)}
                          onClose={() => setOpenTokenId(null)}
                        />
                      </div>
                    )}
                  </span>
                );
              })}
            </div>
          )}
        </div>

        {/* Action buttons */}
        {promptDoc && (
          <div className="flex gap-3 justify-end">
            <Button variant="outline" onClick={reset}>
              Reset
            </Button>
            <Button 
              onClick={copyToClipboard}
              className="bg-gradient-primary hover:shadow-glow transition-all duration-300"
            >
              Copy Prompt
            </Button>
          </div>
        )}

        {/* Token legend */}
        <div className="flex gap-4 flex-wrap text-xs">
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-token-descriptor"></div>
            Descriptor
          </span>
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-token-quality"></div>
            Quality
          </span>
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-token-mood"></div>
            Mood
          </span>
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-token-action"></div>
            Action
          </span>
        </div>
      </div>
    </div>
  );
}