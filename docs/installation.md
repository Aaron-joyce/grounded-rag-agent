# Grounded RAG Engine Installation & Operations Guide

## Overview
This document provides instructions for setting up, installing, and running the Grounded RAG Engine system locally.

## Prerequisites
- **Python:** Python 3.14 or later
- **Package Manager:** `uv` (Fast Python package installer and resolver)
- **API Key:** A valid Google Gemini API Key (`GEMINI_API_KEY`)

## Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/example/grounded-rag.git
   cd grounded-rag
   ```

2. **Install Dependencies:**
   Use `uv` to synchronize dependencies from `uv.lock`:
   ```bash
   uv sync
   ```

3. **Configure Environment Variables:**
   Copy the example environment file and add your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

4. **Run the RAG Engine:**
   Execute `main.py` to index documents in `docs/` and run sample queries:
   ```bash
   uv run main.py
   ```

## Authentication & API Access
- The engine uses the `google-genai` SDK with native function calling.
- Authentication is handled automatically via the `GEMINI_API_KEY` environment variable.
