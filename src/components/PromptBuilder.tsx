import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

// Types
type Suggestion = string;
type ModelType = "sdxl" | "flux";
type TokenRole = "genre" | "subject" | "attributes" | "action" | "setting" | "composition" | "perspective" | "lighting" | "color" | "style" | "mood" | "quality";

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
  enhance(selection: string, modelType: ModelType): Promise<{ prompt: string }>;
  alternatives(prompt: string): Promise<{ tokens: Token[] }>;
};

// Utils
let idCounter = 0;
const uid = () => `token-${Date.now()}-${++idCounter}`;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Extract fallback tokens when API fails - comprehensive extraction across all categories
function extractFallbackTokens(prompt: string): { tokens: Token[] } {
  const foundTokens: Token[] = [];
  const words = prompt.toLowerCase().split(/\s+/);
  const fullPrompt = prompt.toLowerCase();
  
  // Category-specific terms for comprehensive token extraction
  const categoryTerms: Record<TokenRole, string[]> = {
    genre: ["portrait", "landscape", "character", "product", "narrative", "scene", "architecture", "photographic"],
    subject: ["person", "man", "woman", "child", "cat", "dog", "car", "building", "tree", "mountain", "persian", "fluffy"],
    attributes: ["young", "old", "tall", "short", "beautiful", "elegant", "rugged", "sleek", "modern", "vintage", "intense"],
    action: ["running", "jumping", "sitting", "standing", "smiling", "crying", "dancing", "flying", "pose", "expression"],
    setting: ["forest", "city", "beach", "studio", "office", "home", "park", "street", "room", "outdoor", "indoor"],
    composition: ["close-up", "wide shot", "rule of thirds", "centered", "framing", "depth", "leading lines", "symmetry"],
    perspective: ["50mm", "85mm", "wide angle", "telephoto", "macro", "fisheye", "aerial", "ground level", "lens"],
    lighting: ["golden hour", "soft light", "harsh light", "rim light", "backlit", "studio lighting", "natural light", "dramatic"],
    color: ["warm tones", "cool tones", "vibrant", "muted", "monochrome", "pastel", "neon", "earthy"],
    style: ["oil painting", "photography", "digital art", "watercolor", "pencil", "3D render", "CGI", "hyperrealistic"],
    mood: ["dramatic", "peaceful", "energetic", "melancholy", "joyful", "mysterious", "romantic", "melancholic"],
    quality: ["high quality", "8k", "ultra detailed", "sharp focus", "highly detailed", "masterpiece", "intricate", "photorealistic"]
  };
  
  // Extract multi-word terms first for each category
  for (const [role, terms] of Object.entries(categoryTerms)) {
    for (const term of terms) {
      if (fullPrompt.includes(term) && !foundTokens.some(t => t.text === term)) {
        foundTokens.push({
          id: uid(),
          text: term,
          role: role as TokenRole,
          alts: getDefaultAlternatives(term, role as TokenRole)
        });
      }
    }
  }
  
  // Add individual significant words, ensuring category diversity
  const significantWords = words.filter(word => 
    word.length > 3 && !['with', 'from', 'that', 'this', 'they', 'have', 'will', 'been', 'were', 'very', 'much', 'more'].includes(word)
  );
  
  for (const word of significantWords) {
    if (!foundTokens.some(token => token.text.toLowerCase().includes(word)) && foundTokens.length < 12) {
      const role = inferRole(word);
      foundTokens.push({
        id: uid(),
        text: word,
        role,
        alts: getDefaultAlternatives(word, role)
      });
    }
  }
  
  // Ensure we have tokens from different categories
  const roleCount: Record<TokenRole, number> = {
    genre: 0, subject: 0, attributes: 0, action: 0, setting: 0, composition: 0,
    perspective: 0, lighting: 0, color: 0, style: 0, mood: 0, quality: 0
  };
  
  foundTokens.forEach(token => roleCount[token.role]++);
  
  return { tokens: foundTokens.slice(0, 15) };
}

function getDefaultAlternatives(term: string, role?: TokenRole): string[] {
  // Category-specific alternatives
  const categoryAlts: Record<TokenRole, string[]> = {
    genre: ["portrait", "landscape", "character", "product", "narrative", "scene"],
    subject: ["person", "figure", "character", "individual", "model", "subject"],
    attributes: ["elegant", "modern", "vintage", "sleek", "rugged", "refined"],
    action: ["posing", "moving", "gesturing", "expressing", "performing", "demonstrating"],
    setting: ["studio", "outdoor", "indoor", "natural", "urban", "architectural"],
    composition: ["centered", "off-center", "rule of thirds", "symmetrical", "dynamic", "balanced"],
    perspective: ["wide angle", "telephoto", "macro", "aerial", "eye-level", "low angle"],
    lighting: ["soft light", "hard light", "natural light", "studio light", "ambient", "directional"],
    color: ["warm tones", "cool tones", "vibrant", "muted", "monochrome", "saturated"],
    style: ["photographic", "artistic", "cinematic", "documentary", "commercial", "fine art"],
    mood: ["dramatic", "serene", "energetic", "mysterious", "joyful", "contemplative"],
    quality: ["high detail", "sharp focus", "pristine", "professional", "masterful", "exceptional"]
  };
  
  // Term-specific alternatives
  const altMap: Record<string, string[]> = {
    // Quality terms
    "high quality": ["premium", "photoreal", "film-grade", "clean", "polished", "pristine"],
    "ultra detailed": ["intricate", "highly detailed", "fine detail", "micro-detail", "ornate", "textured"],
    "photorealistic": ["lifelike", "realistic", "authentic", "natural", "true-to-life", "believable"],
    "masterpiece": ["work of art", "tour de force", "magnum opus", "classic", "exemplary", "outstanding"],
    
    // Lighting terms
    "studio lighting": ["natural light", "softbox", "rim light", "backlit", "hard light", "volumetric"],
    "dramatic": ["moody", "intense", "striking", "bold", "atmospheric", "cinematic"],
    "golden hour": ["magic hour", "sunset", "sunrise", "warm light", "evening light", "dusk"],
    
    // Style terms
    "photographic": ["realistic", "documentary", "journalistic", "candid", "editorial", "commercial"],
    "hyperrealistic": ["ultra-realistic", "photoreal", "lifelike", "detailed", "precise", "accurate"],
    
    // Composition terms
    "close-up": ["tight shot", "macro", "detail shot", "intimate", "focused", "cropped"],
    "wide shot": ["panoramic", "landscape", "establishing", "full frame", "expansive", "broad"],
    
    // Color terms
    "warm tones": ["golden", "amber", "sunset", "cozy", "inviting", "soft"],
    "cool tones": ["blue", "cyan", "crisp", "fresh", "clinical", "modern"],
    
    // Subject terms
    "persian": ["exotic", "feline", "elegant", "fluffy", "longhaired", "majestic"],
    "fluffy": ["soft", "furry", "plush", "downy", "fuzzy", "voluminous"],
    "intense": ["piercing", "focused", "sharp", "penetrating", "compelling", "captivating"]
  };
  
  // Use role-specific alternatives if role is provided and no specific term match
  if (role && !altMap[term.toLowerCase()]) {
    return categoryAlts[role];
  }
  
  return altMap[term.toLowerCase()] || categoryAlts[role || "quality"] || ["enhanced", "improved", "refined", "elevated", "sophisticated", "polished"];
}

// Token styling configuration
const ROLE_COLORS: Record<TokenRole, string> = {
  genre: "bg-token-descriptor-bg text-token-descriptor border-token-descriptor-border underline decoration-token-descriptor/70",
  subject: "bg-token-quality-bg text-token-quality border-token-quality-border underline decoration-token-quality/70",
  attributes: "bg-token-mood-bg text-token-mood border-token-mood-border underline decoration-token-mood/70",
  action: "bg-token-action-bg text-token-action border-token-action-border underline decoration-token-action/70",
  setting: "bg-token-descriptor-bg text-token-descriptor border-token-descriptor-border underline decoration-token-descriptor/70",
  composition: "bg-token-quality-bg text-token-quality border-token-quality-border underline decoration-token-quality/70",
  perspective: "bg-token-mood-bg text-token-mood border-token-mood-border underline decoration-token-mood/70",
  lighting: "bg-token-action-bg text-token-action border-token-action-border underline decoration-token-action/70",
  color: "bg-token-descriptor-bg text-token-descriptor border-token-descriptor-border underline decoration-token-descriptor/70",
  style: "bg-token-quality-bg text-token-quality border-token-quality-border underline decoration-token-quality/70",
  mood: "bg-token-mood-bg text-token-mood border-token-mood-border underline decoration-token-mood/70",
  quality: "bg-token-action-bg text-token-action border-token-action-border underline decoration-token-action/70",
};

// Expanded keyword-based role inference for comprehensive categorization
const KEYWORD_ROLE: Array<{ pattern: RegExp; role: TokenRole }> = [
  { 
    pattern: /(portrait|landscape|character|product|narrative|scene|architecture|photographic|street|documentary)/gi, 
    role: "genre" 
  },
  { 
    pattern: /(person|man|woman|child|cat|dog|car|building|tree|mountain|ocean|persian|individual|figure|model)/gi, 
    role: "subject" 
  },
  { 
    pattern: /(young|old|tall|short|beautiful|elegant|rugged|sleek|modern|vintage|intense|fluffy|majestic|exotic)/gi, 
    role: "attributes" 
  },
  { 
    pattern: /(running|jumping|sitting|standing|smiling|crying|dancing|flying|posing|gesturing|expression|gaze)/gi, 
    role: "action" 
  },
  { 
    pattern: /(forest|city|beach|studio|office|home|park|street|room|outdoor|indoor|urban|natural|environment)/gi, 
    role: "setting" 
  },
  { 
    pattern: /(close-up|wide shot|rule of thirds|centered|framing|depth|leading lines|symmetry|balance|composition)/gi, 
    role: "composition" 
  },
  { 
    pattern: /(50mm|85mm|wide angle|telephoto|macro|fisheye|aerial|ground level|lens|perspective|viewpoint|angle)/gi, 
    role: "perspective" 
  },
  { 
    pattern: /(golden hour|soft light|harsh light|rim light|backlit|studio lighting|natural light|dramatic|chiaroscuro|ambient)/gi, 
    role: "lighting" 
  },
  { 
    pattern: /(warm tones|cool tones|vibrant|muted|monochrome|pastel|neon|earthy|palette|color|hue|saturation)/gi, 
    role: "color" 
  },
  { 
    pattern: /(oil painting|photography|digital art|watercolor|pencil|3D render|CGI|hyperrealistic|photorealistic|artistic)/gi, 
    role: "style" 
  },
  { 
    pattern: /(dramatic|peaceful|energetic|melancholy|joyful|mysterious|romantic|serene|moody|contemplative|melancholic)/gi, 
    role: "mood" 
  },
  { 
    pattern: /(high quality|8k|ultra detail|sharp focus|highly detailed|masterpiece|intricate|pristine|professional|premium)/gi, 
    role: "quality" 
  },
];

function inferRole(text: string): TokenRole {
  for (const { pattern, role } of KEYWORD_ROLE) {
    if (pattern.test(text)) return role;
  }
  
  // Fallback heuristics
  if (/\b(ly|ic|ous|ful|ish|ive)\b/i.test(text) || /quality|detail|sharp|crisp/i.test(text)) 
    return "quality";
  if (/\b(run|fly|jump|sit|stand|dance|move)\b/i.test(text)) 
    return "action";
  if (/\b(mood|feel|atmosphere|vibe|emotion)\b/i.test(text)) 
    return "mood";
  if (/\b(style|medium|art|paint|photo)\b/i.test(text)) 
    return "style";
  
  return "subject";
}

// Google Gemini API integration
function createGoogleLLM(apiKey: string): LLMApi {
  const baseUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent';
  
  async function callGemini(prompt: string): Promise<string> {
    const response = await fetch(baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
      },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 1000,
        }
      })
    });

    if (!response.ok) {
      let errorMessage = `Google API error: ${response.status}`;
      
      try {
        const errorData = await response.json();
        if (errorData?.error?.message) {
          errorMessage += ` - ${errorData.error.message}`;
          
          // Specific handling for API key issues
          if (errorData.error.message.includes('API key not valid') || 
              errorData.error.message.includes('API_KEY_INVALID')) {
            throw new Error('INVALID_API_KEY');
          }
        }
      } catch (parseError) {
        // If we can't parse the error, just use the status
      }
      
      throw new Error(errorMessage);
    }

    const data = await response.json();
    const responseText = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    
    if (!responseText.trim()) {
      throw new Error('Empty response from Google API');
    }
    
    return responseText;
  }

  return {
    async complete(text: string) {
      const prompt = `You are a creative AI assistant that helps users complete image prompts. Given a partial prompt, provide exactly 3 creative and diverse completions that would make good image generation prompts. 

Return your response as a JSON object with this exact format:
{"suggestions": ["completion 1", "completion 2", "completion 3"]}

User input: "${text}"`;

      try {
        const response = await callGemini(prompt);
        // Clean response - remove code blocks and extra whitespace
        const cleanResponse = response
          .replace(/```json\n?|\n?```/g, '')
          .replace(/```\n?|\n?```/g, '')
          .trim();
        
        const parsed = JSON.parse(cleanResponse);
        return { suggestions: (parsed.suggestions || []).slice(0, 3) };
      } catch (error) {
        console.error('Error in complete:', error);
        return { suggestions: [] };
      }
    },

    async enhance(selection: string, modelType: ModelType) {
      const isSDXL = modelType === 'sdxl';
      
      const prompt = isSDXL 
        ? `Transform this basic prompt into a structured SDXL prompt using this template: "{GENRE} of {SUBJECT}, {KEY ATTRIBUTES}, {ACTION/SETTING}, {COMPOSITION}, {PERSPECTIVE/LENS}, {LIGHTING}, {COLOR}, {STYLE/MEDIUM}, {MOOD}, {QUALITY BOOSTERS}"

Examples:
- Portrait: "portrait of young woman, elegant dress, confident pose, close-up framing, 85mm lens, soft natural light, warm tones, oil painting style, serene mood, highly detailed, sharp focus"
- Landscape: "landscape of mountain valley, snow-capped peaks, golden hour, wide shot, 24mm lens, atmospheric perspective, warm golden light, earthy palette, photographic style, peaceful mood, high dynamic range"

Return JSON format: {"prompt": "structured prompt here"}

Transform: "${selection}"`
        : `Transform this basic prompt into rich descriptive prose for Flux/Qwen. Maximum 150 words. Write natural, flowing description covering subject details, setting, composition, lighting, and style elements.

Return JSON format: {"prompt": "your descriptive prose here"}

Transform: "${selection}"`;

      try {
        const response = await callGemini(prompt);
        let cleanResponse = response
          .replace(/```json\n?|\n?```/g, '')
          .replace(/```\n?|\n?```/g, '')
          .trim();
        
        // Handle potential JSON parsing issues by finding the JSON object
        const jsonMatch = cleanResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          cleanResponse = jsonMatch[0];
        }
        
        // Clean up control characters in the JSON string values
        cleanResponse = cleanResponse.replace(/\\n/g, '\\n').replace(/\\r/g, '\\r').replace(/\\t/g, '\\t');
        
        const parsed = JSON.parse(cleanResponse);
        return { prompt: parsed.prompt || selection };
      } catch (error) {
        console.error('Error in enhance:', error);
        return { prompt: selection };
      }
    },

    async alternatives(prompt: string) {
      const systemPrompt = `Analyze this image generation prompt and identify key tokens that can be replaced with alternatives. Use diverse categories including:

- genre, subject, attributes, action, setting, composition, perspective, lighting, color, style, mood, quality
- emotion, texture, weather, time, material, clothing, expression, architecture, nature, technology
- camera, lens, filter, effect, technique, medium, format, finish, surface, pattern

Return your response as a JSON object with this exact format:
{"tokens": [{"text": "word", "role": "category", "alts": ["alt1", "alt2", "alt3", "alt4", "alt5", "alt6"]}]}

Provide exactly 6 alternatives for each token. Extract 12-20 tokens across different categories. Focus on all replaceable terms including adjectives, nouns, technical terms, and descriptive elements.

Prompt to analyze: "${prompt}"`;

      try {
        const response = await callGemini(systemPrompt);
        // Clean response - remove code blocks and extra whitespace
        const cleanResponse = response
          .replace(/```json\n?|\n?```/g, '')
          .replace(/```\n?|\n?```/g, '')
          .trim();
        
        let parsed;
        try {
          parsed = JSON.parse(cleanResponse);
        } catch (parseError) {
          console.error('JSON parsing failed, trying to extract JSON from response:', cleanResponse);
          // Try to extract JSON from the response if it's embedded in text
          const jsonMatch = cleanResponse.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            parsed = JSON.parse(jsonMatch[0]);
          } else {
            throw parseError;
          }
        }
        
        const tokens = (parsed.tokens || []).map((token: any) => ({
          id: uid(),
          text: token.text,
          role: token.role as TokenRole,
          alts: Array.isArray(token.alts) ? token.alts : []
        }));
        return { tokens };
      } catch (error) {
        console.error('Error in alternatives:', error);
        // Enhanced fallback with comprehensive token extraction
        return extractFallbackTokens(prompt);
      }
    }
  };
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
  
  async enhance(selection, modelType: ModelType) {
    await sleep(300);
    const isSDXL = modelType === 'sdxl';
    
    if (isSDXL) {
      return { 
        prompt: `portrait of ${selection}, elegant features, confident pose, close-up framing, 85mm lens, soft natural light, warm tones, photographic style, serene mood, highly detailed, sharp focus`
      };
    } else {
      return { 
        prompt: `A character portrait featuring ${selection}. The subject displays elegant features with a confident pose, captured in natural lighting. The portrait is framed as a close-up shot using an 85mm lens perspective with shallow depth of field. Soft natural light creates gentle shadows from the left side, while warm golden tones convey a sense of serenity and elegance. The style is photographic realism with fine detail and sharp focus, emphasizing natural skin texture and authentic expression.`
      };
    }
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

// API Key Management Component
interface APIKeyInputProps {
  apiKey: string;
  setApiKey: (key: string) => void;
}

function APIKeyInput({ apiKey, setApiKey }: APIKeyInputProps) {
  const [isEditing, setIsEditing] = useState(!apiKey);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');

  const validateApiKey = (key: string): boolean => {
    // Basic validation for Google API key format
    return key.startsWith('AIza') && key.length >= 35;
  };

  const handleSave = () => {
    if (!draft.trim()) return;
    
    if (!validateApiKey(draft.trim())) {
      setError('Invalid API key format. Google API keys should start with "AIza"');
      return;
    }
    
    setApiKey(draft.trim());
    localStorage.setItem('google-api-key', draft.trim());
    setIsEditing(false);
    setDraft('');
    setError('');
  };

  const handleClear = () => {
    setApiKey('');
    localStorage.removeItem('google-api-key');
    setIsEditing(true);
    setError('');
  };

  const maskedKey = apiKey ? `••••••••••••${apiKey.slice(-4)}` : '';

  return (
    <div className="mb-6 p-4 rounded-xl border bg-card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">Google Gemini API Key</h3>
        {apiKey && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span className="text-xs text-muted-foreground">Connected</span>
          </div>
        )}
      </div>

      {isEditing ? (
        <div className="space-y-3">
          <Input
            type="password"
            placeholder="Enter your Google Gemini API key (starts with AIza...)"
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setError('');
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSave();
              if (e.key === 'Escape') {
                setIsEditing(false);
                setDraft('');
                setError('');
              }
            }}
            className={`font-mono ${error ? 'border-red-500' : ''}`}
          />
          {error && (
            <div className="text-sm text-red-600 flex items-center gap-2">
              ⚠️ {error}
            </div>
          )}
          <div className="flex gap-2">
            <Button onClick={handleSave} size="sm" disabled={!draft.trim()}>
              Save Key
            </Button>
            {apiKey && (
              <Button variant="outline" size="sm" onClick={() => {
                setIsEditing(false);
                setDraft('');
                setError('');
              }}>
                Cancel
              </Button>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            Your API key is stored locally in your browser and never sent to our servers.{' '}
            <a 
              href="https://aistudio.google.com/app/apikey" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Get your free API key here →
            </a>
          </p>
        </div>
      ) : (
        <div className="flex items-center justify-between">
          <span className="font-mono text-sm">{maskedKey}</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setIsEditing(true)}>
              Edit
            </Button>
            <Button variant="outline" size="sm" onClick={handleClear}>
              Remove
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// UI Components
interface HeaderProps {
  live: boolean;
  setLive: (value: boolean) => void;
  modelType: ModelType;
  setModelType: (type: ModelType) => void;
}

function Header({ live, setLive, modelType, setModelType }: HeaderProps) {
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
      
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">Model:</span>
          <div className="flex items-center bg-muted rounded-lg p-1">
            <button
              onClick={() => setModelType('sdxl')}
              className={`px-3 py-1.5 text-sm rounded-md transition-all ${
                modelType === 'sdxl' 
                  ? 'bg-background shadow-sm text-foreground' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              SDXL
            </button>
            <button
              onClick={() => setModelType('flux')}
              className={`px-3 py-1.5 text-sm rounded-md transition-all ${
                modelType === 'flux' 
                  ? 'bg-background shadow-sm text-foreground' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Flux/Qwen
            </button>
          </div>
        </div>
        
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
  const [apiKey, setApiKey] = useState("");
  const [modelType, setModelType] = useState<ModelType>("sdxl");

  const [selection, setSelection] = useState<string | null>(null);
  const [loadingB, setLoadingB] = useState(false);
  const [promptDoc, setPromptDoc] = useState<PromptDoc | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [openTokenId, setOpenTokenId] = useState<string | null>(null);

  // Load API key from localStorage on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('google-api-key');
    if (savedKey) {
      setApiKey(savedKey);
    }
  }, []);

  // Get the appropriate LLM based on API key availability
  const getLLM = (): LLMApi => {
    return apiKey ? createGoogleLLM(apiKey) : MockLLM;
  };

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
        const llm = getLLM();
        const result = await llm.complete(query);
        if (reqId === completeReqId.current) {
          setSuggestions(result.suggestions || []);
          setLoadingA(false);
        }
        } catch (error) {
          if (reqId === completeReqId.current) {
            setLoadingA(false);
            console.error('Error getting suggestions:', error);
            
            // Show user-friendly error for API key issues
            if (error instanceof Error && error.message === 'INVALID_API_KEY') {
              alert('⚠️ API Key Invalid\n\nYour Google API key appears to be invalid or expired. Please check your API key and try again.\n\nMake sure your API key:\n• Is correctly copied\n• Has Gemini API access enabled\n• Hasn\'t expired');
            }
          }
        }
    }, 300);

    return () => clearTimeout(timer);
  }, [input, live, apiKey]);

  const handleSuggest = async () => {
    setOpenTokenId(null);
    setSelection(null);
    setPromptDoc(null);
    setSegments([]);

    if (!input.trim()) return;

    setLoadingA(true);
    try {
      const llm = getLLM();
      const result = await llm.complete(input);
      setSuggestions(result.suggestions || []);
    } catch (error) {
      console.error('Error getting suggestions:', error);
      
      // Show user-friendly error for API key issues
      if (error instanceof Error && error.message === 'INVALID_API_KEY') {
        alert('⚠️ API Key Invalid\n\nYour Google API key appears to be invalid or expired. Please check your API key and try again.\n\nMake sure your API key:\n• Is correctly copied\n• Has Gemini API access enabled\n• Hasn\'t expired');
      }
    } finally {
      setLoadingA(false);
    }
  };

  const pickSuggestion = async (suggestion: string) => {
    setSelection(suggestion);
    setLoadingB(true);

    try {
      const llm = getLLM();
      const enhanced = await llm.enhance(suggestion, modelType);
      let altResult = await llm.alternatives(enhanced.prompt);
      
      // Fallback: if no tokens from API, extract some basic ones from the text
      if (!altResult.tokens || altResult.tokens.length === 0) {
        console.log('No tokens from API, extracting fallback tokens');
        altResult = extractFallbackTokens(enhanced.prompt);
      }
      
      const segments = segmentPrompt(enhanced.prompt, altResult.tokens);
      
      setPromptDoc({ raw: enhanced.prompt, tokens: altResult.tokens });
      setSegments(segments);
    } catch (error) {
      console.error('Error enhancing prompt:', error);
      
      // Show user-friendly error for API key issues
      if (error instanceof Error && error.message === 'INVALID_API_KEY') {
        alert('⚠️ API Key Invalid\n\nYour Google API key appears to be invalid or expired. Please check your API key and try again.\n\nMake sure your API key:\n• Is correctly copied\n• Has Gemini API access enabled\n• Hasn\'t expired');
      } else {
        // Fallback: use the original suggestion and extract basic tokens
        console.log('API failed, using fallback enhancement and tokens');
        const fallbackPrompt = modelType === 'sdxl' 
          ? `portrait of ${suggestion}, detailed features, professional lighting, high quality, sharp focus`
          : `A detailed portrait of ${suggestion}. The subject is captured with professional lighting and careful attention to detail. The composition emphasizes natural beauty with high-quality sharp focus throughout.`;
        const fallbackTokens = extractFallbackTokens(fallbackPrompt);
        const segments = segmentPrompt(fallbackPrompt, fallbackTokens.tokens);
        
        setPromptDoc({ raw: fallbackPrompt, tokens: fallbackTokens.tokens });
        setSegments(segments);
      }
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
      <Header live={live} setLive={setLive} modelType={modelType} setModelType={setModelType} />

      {/* API Key Input */}
      <APIKeyInput apiKey={apiKey} setApiKey={setApiKey} />

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
        <div className="min-h-[24px] text-sm text-muted-foreground flex items-center justify-between">
          <div>
            {loadingA ? (
              <span className="animate-pulse">Finding suggestions...</span>
            ) : suggestions[0] ? (
              <span className="opacity-70">→ {suggestions[0]}</span>
            ) : null}
          </div>
          <div className="text-xs">
            {apiKey ? (
              <span className="text-green-600 flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                Using Gemini API
              </span>
            ) : (
              <span className="text-amber-600 flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-amber-500"></div>
                Demo mode
              </span>
            )}
          </div>
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
        <div className="flex gap-3 flex-wrap text-xs">
          <span className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-token-descriptor"></div>
            Genre/Setting
          </span>
          <span className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-token-quality"></div>
            Subject/Composition
          </span>
          <span className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-token-mood"></div>
            Attributes/Perspective
          </span>
          <span className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-token-action"></div>
            Action/Lighting/Quality
          </span>
        </div>

        {/* Model description */}
        <div className="text-xs text-muted-foreground bg-muted/50 rounded-lg p-3">
          <div className="font-medium mb-1">
            {modelType === 'sdxl' ? '📐 SDXL Template' : '📝 Flux/Qwen Prose'}
          </div>
          <div>
            {modelType === 'sdxl' 
              ? 'Structured format: {GENRE} of {SUBJECT}, {ATTRIBUTES}, {ACTION/SETTING}, {COMPOSITION}, {PERSPECTIVE}, {LIGHTING}, {COLOR}, {STYLE}, {MOOD}, {QUALITY}'
              : 'Natural prose format: Up to 200 words describing purpose, attributes, composition, lighting, and style in flowing sentences.'
            }
          </div>
        </div>
      </div>
    </div>
  );
}