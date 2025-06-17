import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import fs from "fs/promises"; // Use fs.promises for async file operations
import path from "path";
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
import 'dotenv/config'; // Ensure dotenv is configured to load .env files

// --- PDF and environment setup ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Construct the pdfPath in a more straightforward way
const pdfPath = path.join(__dirname, '..', 'data', 'employee_handbook.pdf');

// Retrieve environment variables
const token = process.env["AZURE_INFERENCE_SDK_KEY"];
const endpoint = process.env["AZURE_INFERENCE_SDK_ENDPOINT"];
const modelName = "gpt-4o"; // or your preferred model

// Check if environment variables are set
if (!token || !endpoint) {
  console.error("Error: AZURE_INFERENCE_SDK_KEY or AZURE_INFERENCE_SDK_ENDPOINT is not set in environment variables.");
  // Exit the process or handle this more gracefully based on your application's needs
  process.exit(1);
}

// Initialize the Azure OpenAI client
const client = ModelClient(
  endpoint,
  new AzureKeyCredential(token),
);

const app = express();
app.use(cors());
app.use(bodyParser.json());

// --- PDF chunking helpers ---
let pdfText = null;
let pdfChunks = [];
const CHUNK_SIZE = 2000;

/**
 * Loads the PDF content and chunks it into smaller pieces.
 * Caches the result to avoid reprocessing on every request.
 */
async function loadPDF() {
  if (pdfText) return pdfText; // Return cached text if already loaded

  try {
    // Check if PDF file exists
    await fs.access(pdfPath, fs.constants.F_OK);
  } catch (error) {
    console.error(`Error: PDF not found at ${pdfPath}`);
    return "PDF not found.";
  }

  try {
    const dataBuffer = await fs.readFile(pdfPath); // Use async readFile
    const data = await pdfParse(dataBuffer);
    pdfText = data.text;

    let currentChunk = "";
    // Split by whitespace to process words
    const words = pdfText.split(/\s+/);

    for (const word of words) {
      // Check if adding the next word exceeds the chunk size
      if ((currentChunk + (currentChunk ? " " : "") + word).length <= CHUNK_SIZE) {
        currentChunk += (currentChunk ? " " : "") + word;
      } else {
        // If current chunk is not empty, push it
        if (currentChunk) {
          pdfChunks.push(currentChunk);
        }
        // Start a new chunk with the current word
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
 * Retrieves relevant content chunks from the PDF based on the query.
 * It scores chunks by the number of matching query terms.
 * @param {string} query - The user's query.
 * @returns {string[]} An array of relevant content chunks.
 */
function retrieveRelevantContent(query) {
  if (!pdfChunks.length) {
    console.warn("PDF chunks not available for retrieval. Ensure PDF is loaded.");
    return [];
  }

  // Pre-process query terms for better matching
  const queryTerms = query.toLowerCase()
    .split(/\s+/) // Split by whitespace
    .filter(term => term.length > 2) // Filter out very short terms
    .map(term => term.replace(/[.,?!;:()"']/g, "")); // Remove punctuation

  if (queryTerms.length === 0) return [];

  const scoredChunks = pdfChunks.map(chunk => {
    const chunkLower = chunk.toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      // Create a regex for each term to find all occurrences
      const regex = new RegExp(term, 'gi'); // 'gi' for global, case-insensitive match
      const matches = chunkLower.match(regex);
      if (matches) {
        score += matches.length; // Add the count of matches to the score
      }
    }
    return { chunk, score };
  });

  // Sort by score in descending order and return top 3
  return scoredChunks
    .filter(item => item.score > 0) // Only include chunks with a score
    .sort((a, b) => b.score - a.score)
    .slice(0, 3) // Get top 3 most relevant chunks
    .map(item => item.chunk); // Return only the chunk text
}

// --- RAG-enabled chat endpoint ---
app.post("/chat", async (req, res) => {
  const userMessage = req.body.message;
  // Default useRAG to true if not provided
  const useRAG = req.body.useRAG === undefined ? true : req.body.useRAG;

  let messages = [];
  let sources = [];

  if (useRAG) {
    const pdfLoadResult = await loadPDF(); // Ensure PDF is loaded
    if (pdfLoadResult.startsWith("Error") || pdfLoadResult.startsWith("PDF not found")) {
      // Handle cases where PDF couldn't be loaded
      messages.push({
        role: "system",
        content: "You are a helpful assistant. I apologize, but I could not access the employee handbook to retrieve information at this time."
      });
    } else {
      sources = retrieveRelevantContent(userMessage);

      if (sources.length > 0) {
        // Construct the system message with retrieved sources
        messages.push({
          role: "system",
          content: `You are a helpful assistant answering questions about the company based on its employee handbook.
          Use ONLY the following information from the handbook to answer the user's question.
          If you can't find relevant information in the provided context, state that clearly and politely, without making up information.
          --- EMPLOYEE HANDBOOK EXCERPTS ---
          ${sources.join('\n\n')}
          --- END OF EXCERPTS ---`
        });
      } else {
        // If no relevant sources found, inform the model
        messages.push({
          role: "system",
          content: "You are a helpful assistant. No highly relevant information was found in the employee handbook for this specific question. Please answer generally or state you don't have enough information from the handbook."
        });
      }
    }
  } else {
    // If RAG is disabled, provide a general system prompt
    messages.push({
      role: "system",
      content: "You are a helpful assistant answering questions about the company."
    });
  }

  // Add the user's message to the conversation
  messages.push({ role: "user", content: userMessage });

  try {
    // Call the Azure OpenAI chat completions API
    const response = await client.path("/chat/completions").post({
      body: {
        messages,
        max_tokens: 1000,
        temperature: 1,
        top_p: 1,
        model: modelName
      }
    });
    if (isUnexpected(response)) throw new Error(response.body.error || "Model API error");

    // Add this check:
    if (!response.body.choices || !response.body.choices[0]) {
      console.error("Azure OpenAI response:", response.body);
      throw new Error("No choices returned from the model.");
    }

    res.json({
      reply: response.body.choices[0].message.content,
      sources: useRAG ? sources : []
    });
  } catch (err) {
    console.error("Error calling Azure OpenAI model:", err.message);
    res.status(500).json({ error: "Model call failed", message: err.message });
  }
});

// --- Start server ---
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`AI API server running on port ${PORT}`);
  // Attempt to load PDF on server startup to pre-cache
  loadPDF().then(() => {
    console.log("Initial PDF loading attempt complete.");
  }).catch(err => {
    console.error("Failed initial PDF load:", err);
  });
});
