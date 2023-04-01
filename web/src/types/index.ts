export interface ChatMessage {
    role: "user" | "assistant" | "system";
    type: "text" | "image" | "audio" | "video" | "code";
    first: boolean;
    content: string;
  }

export interface CleanChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface Collection {
  chatgpt: {
    [key: string]: ChatMessage[];
  };
  hugginggpt: {
    [key: string]: ChatMessage[];
  };
}
