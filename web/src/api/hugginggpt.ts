import type { CleanChatMessage } from "@/types";
import axios, { AxiosError } from "axios";
import { HUGGINGGPT_BASE_URL } from "@/config";

const model = "gpt-3.5-turbo";

axios.defaults.headers.post["Content-Type"] = "application/json";

export async function hugginggpt(messageList: CleanChatMessage[]) {
  var endpoint = `${HUGGINGGPT_BASE_URL}/hugginggpt`
  try {
    const response = await axios({
      url: endpoint,
      method: "post",
      data: {
        model,
        messages: messageList.slice(1)
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
      message: error.message
    };
  }
}
