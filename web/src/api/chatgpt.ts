import type { CleanChatMessage } from "@/types";
import axios, { AxiosError } from "axios";
import { CHAT_GPT_URL, CHAT_GPT_LLM } from "@/config";

axios.defaults.headers.post["Content-Type"] = "application/json";

export async function chatgpt(messageList: CleanChatMessage[], apiKey: string) {
  var endpoint = `${CHAT_GPT_URL}/v1/chat/completions`

  try {
    const completion = await axios({
      url: endpoint,
      method: "post",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      data: {
        model: CHAT_GPT_LLM,
        messages: messageList
      },
      timeout: 60000, // 180 seconds
    });
    return {
      status: "success",
      data: completion.data.choices[0].message.content,
    };
  } catch (error: any) {
    return {
      status: "error",
      message: error.message
    };
  }
}
