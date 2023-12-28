from fastapi import FastAPI
from asi_project.pipelines.average.pipeline import pipeline_tests

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to your FastAPI app!"}

@app.get("/pipeline")
async def run_pipeline():
    result = pipeline_tests()  # Replace with your actual pipeline logic
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
