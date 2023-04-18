import type {Collection, ChatMessage } from "@/types";

const ChatGPTTerminalMessage:ChatMessage[] = [
  {
    role: "assistant",
    content: "Hi there! I am OpenAI ChatGPT, an AI assistant for you. How can I help you? ",
    type: "text",
    first: true
  },
  {
    role: "user",
    content: "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. When I need to tell you something in English, I will do so by putting text inside curly brackets {like this}.",
    type: "text",
    first: true
  },
  {
    role: "assistant",
    content: "Yes, I will do it for you. Please type the command and I will reply with the terminal output.",
    type: "text",
    first: true
  }
]

const ChatGPTPolishMessage:ChatMessage[] = [
    {
      role: "assistant",
      content: "Hi there! I am OpenAI ChatGPT, an AI assistant for you. How can I help you? ",
      type: "text",
      first: true
    },
    {
      role: "user",
      content: "You are a well-trained AI writing assistant with expertise in writing academic papers for computer conferences. By giving you a draft paragraph, I hope you can help me polish my writing with your knowledge. The language should be concise and consistent with the style of an academic paper.",
      type: "text",
      first: true
    },
    {
      role: "assistant",
      content: "No problem, I will think carefully and polish the paper for you.",
      type: "text",
      first: true
    },
]

const ChatGPTTranslationMessage:ChatMessage[] = [
  {
    role: "assistant",
    content: "Hi there! I am OpenAI ChatGPT, an AI assistant for you. How can I help you? ",
    type: "text",
    first: true
  },
  {
    role: "user",
    content: "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations.",
    type: "text",
    first: true
  },
  {
    role: "assistant",
    content: "Sure, I will act as an English translator and improver.",
    type: "text",
    first: true
  },
]


const defaultChatGPTMessage:ChatMessage[] = [
  {
    role: "assistant",
    content: "Hi there! I am OpenAI ChatGPT, an AI assistant for you. How can I help you? ",
    type: "text",
    first: true
  }
]
  
const defaultHuggingGPTMessage:ChatMessage[] = [
  {
    role: "assistant",
    content: "Hi there, I am HuggingGPT empowered by Huggingface family! Yes, I can provide thousands of models for dozens of tasks. For more fun and creativity, I have invited Diffusers family to join our team. Feel free to experience it!",
    type: "text",
    first: true
  }
]

const promptCollection: Collection = {
  "chatgpt": {
    "terminal": ChatGPTTerminalMessage,
    "polish": ChatGPTPolishMessage,
    "translation": ChatGPTTranslationMessage,
    "default": defaultChatGPTMessage,
  },
  "hugginggpt": {
    "default": defaultHuggingGPTMessage
  }
}


export default promptCollection