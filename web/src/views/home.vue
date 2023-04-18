<script setup lang="ts">
import type { ChatMessage, CleanChatMessage } from "@/types";
import { ref, watch, nextTick, onMounted, computed } from "vue";
import { RouterLink } from "vue-router";
import { hugginggpt } from "@/api/hugginggpt";
import { chatgpt } from "@/api/chatgpt";
import Loading from "@/components/Loading.vue";
import promptCollection from "@/prompt";
import BASE_URL from "@/config";

let dev = ref(false);
let isChatgpt = ref(false);
let isTalking = ref(false);
let isConfig = ref<boolean>(true);
let title = ref<string>();
let mode = ref<string>("default");

title.value = isChatgpt.value? "ChatGPT": "HuggingGPT";

isConfig.value = (!dev.value && isChatgpt.value)? true : false

const chatListDom = ref<HTMLDivElement>();
// const pdf = ref<HTMLDivElement>();
let messageContent = ref("");

const roleAliasChatHuggingGPT = { user: "Human", assistant: "HuggingGPT", system: "System" };
const roleAliasChatGPT = { user: "Human", assistant: "ChatGPT", system: "System" };
const roleAlias = ref(isChatgpt.value? roleAliasChatGPT: roleAliasChatHuggingGPT);
const messageList = ref<ChatMessage[]>(isChatgpt.value? promptCollection["chatgpt"][mode.value]: promptCollection["hugginggpt"][mode.value]);

onMounted(() => {
  const apiKey = loadConfig();
  if (apiKey) {
    // switchConfigStatus(); //close
    isConfig.value = false
  }
});

async function sendChatMessage() {
  isTalking.value = true;
  const input = messageContent.value
  messageList.value.push(
    { role: "user", content: input, type: "text", first: true},
  )

  clearMessageContent();
  var clean_messages: CleanChatMessage[] = []
  for (let message of messageList.value) {
    if (message.first && message.role != "system") {
      clean_messages.push({role: message.role, content: message.content})
    }
  }
  messageList.value.push(
    { role: "assistant", content: "", type: "text", first: true},
  )
  if (isChatgpt.value) {
    var { status, data } = await chatgpt(clean_messages, loadConfig(), dev.value);
  } else {
    var { status, data } = await hugginggpt(clean_messages, loadConfig(), dev.value);
  }

  messageList.value.pop()
  if (status === "success" ) {
    if (data) {
      messageList.value.push(
        { role: "assistant", content: data, type: "text", first: true }
      );
    } else {
      messageList.value.push(
        { role: "assistant", content: "empty content", type: "text", first: true }
      );
    }
  } else {
    messageList.value.push(
      { role: "system", content: data, type: "text", first: true }
    );
  }
  isTalking.value = false;
}


const messageListMM = computed(() => {
  var messageListMM: ChatMessage[] = []
  for (var i = 0; i < messageList.value.length; i++) {
    var message = messageList.value[i]
    if (message.type != "text") {
      messageListMM.push(message)
      continue
    }
    var content = message.content
    var role = message.role
    
    var image_urls = content.match(/(http(s?):|\/)([/|.|\S||\w|:|-])*?\.(?:jpg|jpeg|tiff|gif|png)/g)
    var image_reg = new RegExp(/(http(s?):|\/)([/|.|\S|\w|:|-])*?\.(?:jpg|jpeg|tiff|gif|png)/g)
    
    var orig_content = content
    var seq_added_accum = 0
    if (image_urls){
      for (var j = 0; j < image_urls.length; j++) {
        // @ts-ignore
        var start = image_reg.exec(orig_content).index
        var end = start + image_urls[j].length
        start += seq_added_accum
        end += seq_added_accum
        const replace_str = `<span class="inline-flex items-baseline">
          <a class="inline-flex text-sky-800 font-bold items-baseline" target="_blank" href="${image_urls[j].startsWith("http")?image_urls[j]:BASE_URL+image_urls[j]}">
              <img src="${image_urls[j].startsWith("http")?image_urls[j]:BASE_URL+image_urls[j]}" alt="" class="inline-flex self-center w-5 h-5 rounded-full mx-1" />
              <span class="mx-1">[Image]</span>
          </a>
          </span>`
        const rep_length = replace_str.length
        seq_added_accum += (rep_length - image_urls[j].length)
        content = content.slice(0, start) + replace_str + content.slice(end)
        
        if(!image_urls[j].startsWith("http")){
          image_urls[j] = BASE_URL + image_urls[j]
        }
      }
    }
  
    orig_content = content
    var audio_urls = content.match(/(http(s?):|\/)([/|.|\w|\S|:|-])*?\.(?:flac|wav)/g)
    var audio_reg = new RegExp(/(http(s?):|\/)([/|.|\w|\S|:|-])*?\.(?:flac|wav)/g)
  
    var seq_added_accum = 0
    if (audio_urls){
      for (var j = 0; j < audio_urls.length; j++) {
        // @ts-ignore
        var start = audio_reg.exec(orig_content).index
        var end = start + audio_urls[j].length
        start += seq_added_accum
        end += seq_added_accum
        const replace_str = `<span class="inline-flex items-baseline">
            <a class="text-sky-800 inline-flex font-bold items-baseline" target="_blank" href="${audio_urls[j].startsWith("http")?audio_urls[j]:BASE_URL+audio_urls[j]}">
              <img class="inline-flex self-center w-5 h-5 rounded-full mx-1" src="/audio.svg"/>
              <span class="mx-1">[Audio]</span>
            </a>
          </span>`
        const rep_length = replace_str.length
        seq_added_accum += (rep_length - audio_urls[j].length)
        content = content.slice(0, start) + replace_str + content.slice(end)
        
        if(!audio_urls[j].startsWith("http")){
          audio_urls[j] = BASE_URL + audio_urls[j]
        }
      }
    }

    orig_content = content
    var video_urls = content.match(/(http(s?):|\/)([/|.|\w|\s|:|-])*?\.(?:mp4)/g)
    var video_reg = new RegExp(/(http(s?):|\/)([/|.|\w|\s|:|-])*?\.(?:mp4)/g)
  
    var seq_added_accum = 0
    if (video_urls){
      for (var j = 0; j < video_urls.length; j++) {
        // @ts-ignore
        var start = video_reg.exec(orig_content).index
        var end = start + video_urls[j].length
        start += seq_added_accum
        end += seq_added_accum
        const replace_str = `<span class="inline-flex items-baseline">
            <a class="text-sky-800 inline-flex font-bold items-baseline" target="_blank" href="${video_urls[j].startsWith("http")?video_urls[j]:BASE_URL+video_urls[j]}">
              <img class="inline-flex self-center w-5 h-5 rounded-full mx-1" src="/video.svg"/>
              <span class="mx-1">[video]</span>
            </a>
          </span>`
        const rep_length = replace_str.length
        seq_added_accum += (rep_length - video_urls[j].length)
        content = content.slice(0, start) + replace_str + content.slice(end)
        
        if(!video_urls[j].startsWith("http")){
          video_urls[j] = BASE_URL + video_urls[j]
        }
      }
    }

    message = {role: role, content: content, type: "text", first: true}
    messageListMM.push(message)
    // de-depulicate
    // @ts-ignore
    image_urls = [...new Set(image_urls)]
    // @ts-ignore
    audio_urls = [...new Set(audio_urls)]
    // @ts-ignore
    video_urls = [...new Set(video_urls)]
    if (image_urls) {
      
      for (var j = 0; j < image_urls.length; j++) {
        messageListMM.push({role: role, content: image_urls[j], type: "image", first: false})
      }
    }
    if (audio_urls) {
      for (var j = 0; j < audio_urls.length; j++) {
        messageListMM.push({role: role, content: audio_urls[j], type: "audio", first: false})
      }
    }
    if (video_urls) {
      for (var j = 0; j < video_urls.length; j++) {
        messageListMM.push({role: role, content: video_urls[j], type: "video", first: false})
      }
    }
    // if (code_blocks){
    //   for (var j = 0; j < code_blocks.length; j++) {
    //     messageListMM.push({role: role, content: code_blocks[j], type: "code", first: false})
    //   }
    // }
  }
  // nextTick(()=>scrollToBottom())
  return messageListMM
})

const sendOrSave = () => {
  if (!messageContent.value.length) return;
  if (isConfig.value) {
    if (saveConfig(messageContent.value.trim())) {
      switchConfigStatus();
    }
    clearMessageContent();
  } else {
    sendChatMessage();
  }
};

const clickConfig = () => {
  if (!isConfig.value) {
    messageContent.value = loadConfig();
  } else {
    clearMessageContent();
  }
  switchConfigStatus();
};


const switchChatGPT = () => {
  isChatgpt.value = !isChatgpt.value;
  if (isChatgpt.value) {
    title.value = "ChatGPT"
    roleAlias.value = roleAliasChatGPT
  } else {
    title.value = "HuggingGPT"
    roleAlias.value = roleAliasChatHuggingGPT
  }
};

function saveConfig(apiKey: string) {
  if (apiKey.slice(0, 3) !== "sk-" || apiKey.length !== 51) {
    alert("Illegal API Key");
    return false;
  }
  localStorage.setItem("apiKey", apiKey);
  return true;
}

function loadConfig() {
  return localStorage.getItem("apiKey") ?? "";
}

function scrollToBottom() {
  if (!chatListDom.value) return;
  // scrollTo(0, chatListDom.value.scrollHeight);
  chatListDom.value.scrollIntoView(false);
}

function switchConfigStatus() {
  isConfig.value = !isConfig.value;
}

function clearMessageContent() {
  messageContent.value = "";
}

// const generateScreenshot = async ()=>{
//   const canvas = await html2canvas(pdf.value)
//   let a = new jsPDF("p", "mm", "a4")
//   // 
//   a.addImage(canvas.toDataURL("image/png"), "PNG", 0, 0, 211, 298);
//   a.save("screenshot.pdf")
// }

watch(mode, ()=> {
  if (isChatgpt.value) {
    messageList.value = promptCollection["chatgpt"][mode.value]
  } else {
    messageList.value = promptCollection["hugginggpt"][mode.value]
  }
})

watch(isChatgpt, () => {
  if (isChatgpt.value) {
    mode.value = "default"
    messageList.value = promptCollection["chatgpt"]["default"]
  } else {
    mode.value = "default"
    messageList.value = promptCollection["hugginggpt"]["default"]
  }
});

// messageList -> messageListMM
watch(messageListMM, () => nextTick(() => {
  nextTick(()=>scrollToBottom())
  }));
</script>

<template>
  <div class="flex flex-row justify-center verflow-auto">
    <!-- <button @click="generateScreenshot">Generate Screenshot</button> -->
  <div class="flex flex-col h-screen max-w-lg border-x-2 border-slate-200">
    
    
    <div class="flex flex-col h-20">
      <div class="flex flex-nowrap fixed max-w-lg w-full items-center justify-between top-0 px-6 py-6 bg-gray-100 z-50 h-20">
        <div class="font-bold w-1/4">
          <!-- <img src="@/assets/chatgpt.svg" class="w-7 mr-1 inline"/>
          x
          <img src="@/assets/huggingface.svg" class="w-8 ml-1  inline"/> -->
          <img src="@/assets/logo.svg" class="w-24 ml-1  inline"/>
        </div>

        <div class="text-2xl font-bold w-1/2 flex justify-center">
          <RouterLink to="/">{{title}}</RouterLink>
        </div>
        
        <div class="text-sm cursor-pointer w-1/4 flex flex-row justify-end" @click="dev || clickConfig()" @dblclick="switchChatGPT()">
            <img src="@/assets/setting.svg" class="w-7 block" title="click to switch to configuration OpenAI key or double click to switch HuggingGPT and ChatGPT"/>
        </div>
      </div>
    </div>

    <div class="flex-1 overflow-auto"  ref="pdf">
      <div class="m-5" ref="chatListDom">
        <div class="relative border-2 rounded-xl p-3" :class="{'bg-violet-50':item.role=='user', 'bg-blue-50':item.role=='assistant', 'bg-yellow-50':item.role=='system', 'mt-4': item.first, 'mt-1': !item.first }" v-for="item of messageListMM" >
          <svg xmlns="http://www.w3.org/2000/svg" v-if="!item.first"  fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 absolute -top-4 left-4 stroke-slate-400">
            <path stroke-linecap="round" stroke-linejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 002.112 2.13" />
          </svg>
          <div v-if="item.first" class="font-bold text-sm mb-3 inline">{{roleAlias[item.role]}} :</div>
          <span
            class="text-sm text-slate-600 whitespace-pre-wrap leading-relaxed"
            v-if="item.content && item.type === 'text'"
            ><div class="break-words" v-html="item.content" ></div>
          </span>
          <img
            class="text-sm text-slate-600 whitespace-pre-wrap leading-relaxed"
            v-else-if="item.content && item.type === 'image'" :src="item.content"
          />
          <audio controls class="w-full text-blue-100" v-else-if="item.content && item.type === 'audio'" :src="item.content">
            </audio>

          <video class="w-full" v-else-if="item.content && item.type === 'video'" controls>
              <source :src="item.content" type="video/mp4">
          </video>

          <pre class="" v-else-if="item.content && item.type === 'code'">
            <code>
              {{item.content}}
            </code>
          </pre>
          
          <Loading class="mt-2" v-else />
        </div>

      </div>
    </div>

    <div class="sticky bottom-0 w-full p-3 bg-gray-100">
      <div class="-mt-2 m-1 text-sm text-gray-500" v-if="isConfig">
        Please input OpenAI key:
      </div>
      <div class="flex">
        <textarea
          rows="2"
          style="resize:none"
          class="input"
          type="text"
          :placeholder="isConfig ? 'sk-xxxxxxxxxx' : 'Input your message'"
          v-model="messageContent"
          @keydown.enter.prevent="isTalking || sendOrSave()"
        >
        </textarea>
        <!-- <input
          class="input"
          type="text"
          :placeholder="isConfig ? 'sk-xxxxxxxxxx' : 'Input your message'"
          v-model="messageContent"
          @keydown.enter="isTalking || sendOrSave()"
        /> -->
        <div class="flex flex-col justify-center">
        <select v-model="mode"  class="text-sm input w-20 m-1 h-7 p-1">
          <option :selected="m=='default'" v-for="m in Object.keys(promptCollection[isChatgpt?'chatgpt':'hugginggpt'])">{{m}}</option>
        </select>

        <button
          class="btn bg-green-700 hover:bg-green-800 disabled:bg-green-400 focus:bg-green-800 text-sm w-20 m-1 h-7 p-1"
          :disabled="!messageList[messageList.length - 1].content"
          @click="sendOrSave()"
        >
          {{ isConfig ? "Save" : "Submit" }}
        </button>
      </div>

      </div>
    </div>
  </div>
</div>
</template>

<style scoped>
pre {
  font-family: -apple-system, "Noto Sans", "Helvetica Neue", Helvetica,
    "Nimbus Sans L", Arial, "Liberation Sans", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN",
    "Microsoft YaHei", "Wenquanyi Micro Hei", "WenQuanYi Zen Hei", "ST Heiti",
    SimHei, "WenQuanYi Zen Hei Sharp", sans-serif;
}
audio {
  width: 100%;
  background-color: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 0.25rem;
  padding: 0.25rem;
  margin: 0;
}

::-webkit-scrollbar {
/*隐藏滚轮*/
display: none;
}

</style>
