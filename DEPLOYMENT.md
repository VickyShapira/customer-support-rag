# Deploying to Streamlit Community Cloud

This guide will help you deploy your Banking RAG demo to Streamlit Community Cloud for free.

## Prerequisites

- Your code is already on GitHub ✅ (https://github.com/VickyShapira/customer-support-rag)
- You have an OpenAI API key
- You have the vector database files in `data/vector_db/`

## Step-by-Step Deployment

### 1. Ensure Required Files are Committed

Make sure these files are in your repository:
- `app/demo.py` - Your Streamlit app
- `requirements.txt` - Python dependencies
- `src/` - Your RAG pipeline code
- `data/vector_db/` - Your vector database (needed for the app to work)

**IMPORTANT:** If `data/` is in your `.gitignore`, you need to either:
- Remove `data/` from `.gitignore` and commit the vector database, OR
- Generate the vector database on first run (requires modifying the app)

### 2. Sign Up for Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Click "Sign up" and use your GitHub account
3. Authorize Streamlit to access your GitHub repositories

### 3. Deploy Your App

1. Click "New app" in Streamlit Cloud dashboard
2. Select your repository: `VickyShapira/customer-support-rag`
3. Set the branch: `main`
4. Set the main file path: `app/demo.py`
5. Click "Advanced settings" (optional but recommended):
   - Python version: 3.11 (or your version)
   - You can leave other settings as default

### 4. Configure Secrets (API Keys)

Before deploying, you MUST add your OpenAI API key:

1. In the deployment settings, scroll to "Secrets"
2. Add your secrets in TOML format:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

3. Click "Save"

### 5. Deploy!

Click "Deploy" and wait for the app to build and launch (usually 2-5 minutes).

## Post-Deployment

### Your App URL
Once deployed, your app will be available at:
```
https://[your-app-name].streamlit.app
```

### Custom Domain (Optional)
You can customize the URL in Settings → General → App URL

### Managing Your App

- **View logs**: Click "Manage app" → "Logs" to see runtime logs
- **Reboot**: If the app crashes, use "Reboot app"
- **Redeploy**: Push changes to GitHub, and the app auto-updates
- **Delete**: Settings → Delete app

## Troubleshooting

### App Won't Start

**Error: "ModuleNotFoundError"**
- Make sure all dependencies are in `requirements.txt`
- Check that the Python version matches your local environment

**Error: "FileNotFoundError" for vector_db**
- Ensure `data/vector_db/` is committed to your repo
- Check that the path in `demo.py` is correct

**Error: "OpenAI API Error"**
- Verify your API key in Streamlit secrets
- Check that your OpenAI account has credits

### App is Slow

- Streamlit Community Cloud has limited resources (1 CPU, 1GB RAM)
- Consider reducing the number of results or using a smaller model
- Use `@st.cache_resource` for expensive operations (already implemented)

### Data Directory Issues

If your vector database is too large or gitignored:

**Option 1: Use Git LFS for large files**
```bash
git lfs install
git lfs track "data/vector_db/*.parquet"
git lfs track "data/vector_db/*.sqlite3"
git add .gitattributes
git commit -m "Add Git LFS for vector database"
```

**Option 2: Build vector DB on first run**
Modify `demo.py` to check if vector_db exists, and if not, build it from source data.

## Cost Considerations

- **Streamlit Cloud**: FREE for public apps (limited resources)
- **OpenAI API**: You pay per request
  - The demo limits free chat to 5 queries per session to prevent abuse
  - Guided scenario uses ~7 queries
  - Cost: ~$0.001-0.01 per session with GPT-4o-mini

### Protecting Against API Abuse

The demo already includes:
- Free chat limit (5 queries per session)
- No auto-restart on errors
- Session-based rate limiting

For production, consider:
- Adding user authentication
- Implementing rate limiting by IP
- Setting daily API budget limits in OpenAI dashboard

## Monitoring

Check your usage:
- **OpenAI**: https://platform.openai.com/usage
- **Streamlit**: App analytics in your dashboard

## Need Help?

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit Forum: https://discuss.streamlit.io
- GitHub Issues: https://github.com/VickyShapira/customer-support-rag/issues

## Quick Reference

**Local Development:**
```bash
streamlit run app/demo.py
```

**Environment Variables:**
- Local: `.env` file or `.streamlit/secrets.toml`
- Cloud: Streamlit Cloud Secrets (TOML format)

**File Paths:**
- Use relative paths from project root
- Check paths work both locally and in cloud environment
