import type { CleanChatMessage } from "@/types";
import axios, { AxiosError } from "axios";

const model = "gpt-3.5-turbo";

axios.defaults.headers.post["Content-Type"] = "application/json";

export async function hugginggpt(
  messageList: CleanChatMessage[],
  apiKey: string,
  dev: boolean
) {
  var endpoint = "http://localhost:8004/hugginggpt";
  try {
    const response = await axios({
      url: endpoint,
      method: "post",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      data: {
        model,
        messages: messageList,
      },
      timeout: 180000, // 180 seconds
    });
    return {
      status: "success",
      data: response.data.message,
    };
  } catch (error: any) {
    return {
      status: "error",
      message: "Unknown Error, please retry.",
    };
  }
}
