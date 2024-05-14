FROM python

# Set working directory
WORKDIR /app

# Copy the Python script and requirements.txt file
COPY newerversion.py .
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

RUN pip install openai==0.27.9
RUN pip install youtube-transcript-api==0.6.2
RUN pip install tiktoken==0.6.0
RUN pip install faiss-cpu==1.8.0

# Command to run the script
ENTRYPOINT ["python", "newerversion.py"]
