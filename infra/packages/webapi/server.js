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
import 'dotenv/config'; // Ensures environment variables from .env are loaded
import axios from 'axios';

// Get __filename and __dirname for ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define the path to your PDF file. Make sure this path is correct relative to your server.js
// It assumes the PDF is in a 'data' folder one level up from where server.js is.
const pdfPath = path.join(__dirname, '..', 'data', 'employee_handbook.pdf');

// --- Environment Variable Checks ---
// Ensure all necessary environment variables are set
const azureOpenAIApiKey = process.env.AZURE_INFERENCE_SDK_KEY;
const azureInstanceName = process.env.INSTANCE_NAME;
const azureDeploymentName = process.env.DEPLOYMENT_NAME; // e.g., "gpt-4o"
const azureOpenAIApiVersion = "2024-08-01-preview"; // Or your specific API version
const azureBasePath = process.env.AZURE_INFERENCE_SDK_ENDPOINT;
const tavilyApiKey = process.env.TAVILY_API_KEY;

if (!azureOpenAIApiKey || !azureInstanceName || !azureDeploymentName || !azureBasePath || !tavilyApiKey) {
  console.error("Error: One or more required environment variables are not set.");
  console.error("Please ensure AZURE_INFERENCE_SDK_KEY, INSTANCE_NAME, DEPLOYMENT_NAME, AZURE_INFERENCE_SDK_ENDPOINT, and TAVILY_API_KEY are configured in your .env file.");
  process.exit(1); // Exit if critical variables are missing
}

// Initialize AzureChatOpenAI model
const chatModel = new AzureChatOpenAI({
  azureOpenAIApiKey: azureOpenAIApiKey,
  azureOpenAIApiInstanceName: azureInstanceName,
  azureOpenAIApiDeploymentName: azureDeploymentName,
  azureOpenAIApiVersion: azureOpenAIApiVersion,
  basePath: azureBasePath,
  temperature: 1, // Controls randomness of the output
  maxTokens: 4096, // Maximum number of tokens to generate
});

// Initialize Express app and middleware
const app = express();
app.use(cors()); // Enable CORS for cross-origin requests from your React app
app.use(bodyParser.json()); // Parse JSON request bodies

let pdfText = null; // Stores the full text of the PDF
let pdfChunks = []; // Stores text chunks for RAG
const CHUNK_SIZE = 2000; // Size of each text chunk

/**
 * Loads and parses the PDF document, then splits it into manageable chunks.
 * This function is called once when the server starts.
 */
async function loadPDF() {
  if (pdfText) return pdfText; // Return if PDF is already loaded

  try {
    // Check if the PDF file exists at the specified path
    await fs.access(pdfPath, fs.constants.F_OK);
  } catch (error) {
    console.error(`Error: PDF not found at ${pdfPath}. Please ensure 'employee_handbook.pdf' is in the 'data' directory.`);
    return "PDF not found."; // Return an error message
  }

  try {
    // Read the PDF file as a buffer
    const dataBuffer = await fs.readFile(pdfPath);
    // Parse the PDF buffer to extract text
    const data = await pdfParse(dataBuffer);
    pdfText = data.text; // Store the full text

    let currentChunk = "";
    // Split the text into words and iterate to create chunks
    const words = pdfText.split(/\s+/);

    for (const word of words) {
      // If adding the next word keeps the chunk within CHUNK_SIZE, add it
      if ((currentChunk + (currentChunk ? " " : "") + word).length <= CHUNK_SIZE) {
        currentChunk += (currentChunk ? " " : "") + word;
      } else {
        // Otherwise, push the current chunk and start a new one with the current word
        if (currentChunk) {
          pdfChunks.push(currentChunk);
        }
        currentChunk = word;
      }
    }
    // Add the last chunk if it's not empty
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

/**
 * Retrieves relevant content chunks from the loaded PDF based on a query.
 * It performs a simple keyword-based scoring.
 * @param {string} query - The user's query string.
 * @returns {string[]} An array of relevant text chunks.
 */
function retrieveRelevantContent(query) {
  if (!pdfChunks.length) {
    console.warn("PDF chunks not available for retrieval. Ensure PDF is loaded.");
    return [];
  }

  // Preprocess the query to extract relevant terms
  const queryTerms = query.toLowerCase()
    .split(/\s+/) // Split by whitespace
    .filter(term => term.length > 2) // Filter out short terms (e.g., "a", "is")
    .map(term => term.replace(/[.,?!;:()"']/g, "")); // Remove punctuation

  if (queryTerms.length === 0) return []; // If no valid terms, return empty

  // Score each chunk based on the number of query term occurrences
  const scoredChunks = pdfChunks.map(chunk => {
    const chunkLower = chunk.toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      // Use a regular expression to find all matches for the term (case-insensitive)
      const regex = new RegExp(term, 'gi');
      const matches = chunkLower.match(regex);
      if (matches) {
        score += matches.length; // Add the count of matches to the score
      }
    }
    return { chunk, score };
  });

  // Filter out chunks with zero score, sort by score in descending order, and take top 3
  return scoredChunks
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3) // Limit to top 3 most relevant chunks
    .map(item => item.chunk); // Return only the text content of the chunks
}

/**
 * Helper to query Tavily web context for additional information.
 * @param {string} query - The query to send to Tavily.
 * @returns {Promise<string|null>} The answer snippet from Tavily, or null if an error occurs.
 */
async function queryTavily(query) {
  try {
    const res = await axios.post(
      'https://api.tavily.com/search',
      {
        query,
        api_key: tavilyApiKey, // Use the environment variable
        include_answer: true, // Request an answer snippet
        search_depth: "basic" // Use basic search depth for faster results
      },
      { headers: { 'Content-Type': 'application/json' } }
    );
    // Return the answer snippet if available, otherwise null
    return res.data.answer || null;
  } catch (err) {
    console.error("Tavily error:", err.message);
    return null;
  }
}

// Store session histories to maintain separate chat conversations for different users
const sessionMemories = {};

/**
 * Retrieves or creates a Langchain BufferMemory instance for a given session ID.
 * @param {string} sessionId - The unique identifier for the chat session.
 * @returns {BufferMemory} The memory instance for the session.
 */
function getSessionMemory(sessionId) {
  if (!sessionMemories[sessionId]) {
    const history = new ChatMessageHistory(); // Create a new chat message history
    sessionMemories[sessionId] = new BufferMemory({
      chatHistory: history,
      returnMessages: true, // Return messages in Langchain's message format
      memoryKey: "chat_history", // Key to store chat history in memory variables
    });
  }
  return sessionMemories[sessionId];
}

// --- API Routes ---

/**
 * Main chat endpoint for the RAG chatbot.
 * Handles user messages, performs RAG if enabled, and gets responses from the LLM.
 */
app.post("/chat", async (req, res) => {
  const userMessage = req.body.message;
  // Determine if RAG should be used, defaults to true
  const useRAG = req.body.useRAG === undefined ? true : req.body.useRAG;
  const sessionId = req.body.sessionId || "default"; // Use a default session ID if none is provided

  let sources = []; // Array to store retrieved sources

  const memory = getSessionMemory(sessionId); // Get memory for the current session
  const memoryVars = await memory.loadMemoryVariables({}); // Load existing chat history

  if (useRAG) {
    // Retrieve relevant content from the PDF
    sources = retrieveRelevantContent(userMessage);

    // Query Tavily for additional web context
    const tavilySnippet = await queryTavily(userMessage);
    if (tavilySnippet) {
      sources.push(`(From Tavily Search)\n${tavilySnippet}`);
    }
  }

  // Prepare the system message based on whether RAG is used and if sources are found
  const systemMessage = useRAG
    ? {
        role: "system",
        content: sources.length > 0
          ? `You are a helpful assistant for Contoso Electronics. You must ONLY use the information provided below to answer. If the information is not sufficient, state that you cannot answer based on the provided data.

--- EMPLOYEE HANDBOOK EXCERPTS ---
${sources.join('\n\n')}
--- END OF EXCERPTS ---`
          : `You are a helpful assistant for Contoso Electronics. The provided excerpts do not contain relevant information for this question. Reply politely: "I'm sorry, I don't know. The employee handbook and available search results do not contain information about that."`,
      }
    : {
        // If RAG is not used, act as a general helpful assistant
        role: "system",
        content: "You are a helpful and knowledgeable assistant. Answer the user's questions concisely and informatively.",
      };

  try {
    // Build the final messages array for the LLM
    const messages = [
      systemMessage,
      ...(memoryVars.chat_history || []), // Include previous chat history
      { role: "user", content: userMessage }, // Add the current user message
    ];

    // Invoke the chat model with the constructed messages
    const response = await chatModel.invoke(messages);

    // Save the current interaction to memory
    await memory.saveContext({ input: userMessage }, { output: response.content });

    // Send the AI's reply and any sources back to the client
    res.json({ reply: response.content, sources });
  } catch (err) {
    console.error("Error during model invocation:", err);
    res.status(500).json({
      error: "Model call failed",
      message: err.message,
      reply: "Sorry, I encountered an error. Please try again.",
      sources: [] // Ensure sources array is always present in error response
    });
  }
});

/**
 * An endpoint for direct Tavily search.
 * (Note: The current React app calls `queryTavily` directly within the /chat route,
 * so this endpoint might be for future expansion or direct tool usage).
 */
app.post('/tools/search_tavily', async (req, res) => {
  const { query } = req.body.parameters;
  console.log("[Tavily route hit] Query received:", query);
  const result = await queryTavily(query); // Call the helper function
  console.log("[Tavily result]:", result);
  res.json({ result });
});


// Define the port for the server to listen on
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`AI API server running on port ${PORT}`);
  // Attempt to load the PDF when the server starts
  loadPDF()
    .then(() => {
      console.log("Initial PDF loading attempt complete.");
    })
    .catch((err) => {
      console.error("Failed initial PDF load:", err);
    });
});