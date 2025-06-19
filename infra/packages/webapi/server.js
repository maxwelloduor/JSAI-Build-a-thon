import { BufferMemory } from "langchain/memory";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import fs from "fs/promises"; 
import path from "path";
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import { AzureChatOpenAI } from "@langchain/openai";
import 'dotenv/config'; 


const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);


const pdfPath = path.join(__dirname, '..', 'data', 'employee_handbook.pdf');


const token = process.env["AZURE_INFERENCE_SDK_KEY"];
const endpoint = process.env["AZURE_INFERENCE_SDK_ENDPOINT"];
const modelName = "gpt-4o"; // or your preferred model


if (!token || !endpoint) {
  console.error("Error: AZURE_INFERENCE_SDK_KEY or AZURE_INFERENCE_SDK_ENDPOINT is not set in environment variables.");
  
  process.exit(1);
}


const chatModel = new AzureChatOpenAI({
  azureOpenAIApiKey: process.env.AZURE_INFERENCE_SDK_KEY,
  azureOpenAIApiInstanceName: process.env.INSTANCE_NAME, // In target url: https://<INSTANCE_NAME>.services...
  azureOpenAIApiDeploymentName: process.env.DEPLOYMENT_NAME, // i.e "gpt-4o"
  azureOpenAIApiVersion: "2024-08-01-preview",
  basePath: process.env.AZURE_INFERENCE_SDK_ENDPOINT, 
  temperature: 1,
  maxTokens: 4096,
});

const app = express();
app.use(cors());
app.use(bodyParser.json());


let pdfText = null;
let pdfChunks = [];
const CHUNK_SIZE = 2000;


async function loadPDF() {
  if (pdfText) return pdfText; 

  try {
    
    await fs.access(pdfPath, fs.constants.F_OK);
  } catch (error) {
    console.error(`Error: PDF not found at ${pdfPath}`);
    return "PDF not found.";
  }

  try {
    const dataBuffer = await fs.readFile(pdfPath); 
    const data = await pdfParse(dataBuffer);
    pdfText = data.text;

    let currentChunk = "";
    
    const words = pdfText.split(/\s+/);

    for (const word of words) {
      
      if ((currentChunk + (currentChunk ? " " : "") + word).length <= CHUNK_SIZE) {
        currentChunk += (currentChunk ? " " : "") + word;
      } else {
        
        if (currentChunk) {
          pdfChunks.push(currentChunk);
        }
        
        currentChunk = word;
      }
    }
    
    if (currentChunk) {
      pdfChunks.push(currentChunk);
    }

    console.log(`PDF loaded and chunked. Total chunks: ${pdfChunks.length}`);
    return pdfText;
  } catch (error) {
    console.error("Error loading or parsing PDF:", error);
    return "Error processing PDF.";
  }
}


function retrieveRelevantContent(query) {
  if (!pdfChunks.length) {
    console.warn("PDF chunks not available for retrieval. Ensure PDF is loaded.");
    return [];
  }

  
  const queryTerms = query.toLowerCase()
    .split(/\s+/) 
    .filter(term => term.length > 2) 
    .map(term => term.replace(/[.,?!;:()"']/g, "")); 

  if (queryTerms.length === 0) return [];

  const scoredChunks = pdfChunks.map(chunk => {
    const chunkLower = chunk.toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      
      const regex = new RegExp(term, 'gi'); 
      const matches = chunkLower.match(regex);
      if (matches) {
        score += matches.length; 
      }
    }
    return { chunk, score };
  });

  
  return scoredChunks
    .filter(item => item.score > 0) 
    .sort((a, b) => b.score - a.score)
    .slice(0, 3) 
    .map(item => item.chunk); 
}


app.post("/chat", async (req, res) => {
  const userMessage = req.body.message;
  const useRAG = req.body.useRAG === undefined ? true : req.body.useRAG;
  const sessionId = req.body.sessionId || "default";

  let sources = [];

  const memory = getSessionMemory(sessionId);
  const memoryVars = await memory.loadMemoryVariables({});

  if (useRAG) {
    await loadPDF();
    sources = retrieveRelevantContent(userMessage);
  }

  // Prepare system prompt
  const systemMessage = useRAG
    ? {
        role: "system",
        content: sources.length > 0
          ? `You are a helpful assistant for Contoso Electronics. You must ONLY use the information provided below to answer.

--- EMPLOYEE HANDBOOK EXCERPTS ---
${sources.join('\n\n')}
--- END OF EXCERPTS ---`
          : `You are a helpful assistant for Contoso Electronics. The excerpts do not contain relevant information for this question. Reply politely: "I'm sorry, I don't know. The employee handbook does not contain information about that."`,
      }
    : {
        role: "system",
        content: "You are a helpful and knowledgeable assistant. Answer the user's questions concisely and informatively.",
      };

  try {
    // Build final messages array
    const messages = [
      systemMessage,
      ...(memoryVars.chat_history || []),
      { role: "user", content: userMessage },
    ];

    const response = await chatModel.invoke(messages);

    await memory.saveContext({ input: userMessage }, { output: response.content });

    res.json({ reply: response.content, sources });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Model call failed",
      message: err.message,
      reply: "Sorry, I encountered an error. Please try again."
    });
  }
});


// Store session histories, allowing you to maintain separate chat histories for different users or sessions.
const sessionMemories = {};

function getSessionMemory(sessionId) {
  if (!sessionMemories[sessionId]) {
    const history = new ChatMessageHistory();
    sessionMemories[sessionId] = new BufferMemory({
      chatHistory: history,
      returnMessages: true,
      memoryKey: "chat_history",
    });
  }
  return sessionMemories[sessionId];
}


const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`AI API server running on port ${PORT}`);
  loadPDF().then(() => {
    console.log("Initial PDF loading attempt complete.");
  }).catch(err => {
    console.error("Failed initial PDF load:", err);
  });
});
