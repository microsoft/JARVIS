import type { CleanChatMessage } from "@/types";
import axios, { AxiosError } from "axios";

const model = "gpt-3.5-turbo";
// const model = "text-davinci-003"

axios.defaults.headers.post["Content-Type"] = "application/json";

export async function chatgpt(messageList: CleanChatMessage[], apiKey: string, dev: boolean) {
  if (dev) {
    var endpoint = "http://localhost:8003/v1/chat/completions"
  } else {
    var endpoint = "https://api.openai.com/v1/chat/completions"
  }

  try {
    const completion = await axios({
      url: endpoint,
      method: "post",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      data: {
        model,
        messages: messageList
      },
    });
    return {
      status: "success",
      data: completion.data.choices[0].message.content,
    };
  } catch (error: any) {
    return {
      status: "error",
      data: "Something seems wrong"
    };
  }
}
