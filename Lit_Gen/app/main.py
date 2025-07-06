
import pandas as pd
import os

def load_reddit_data(directory="."):
    posts_data = []
    comments_data = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.startswith("reddit_posts_") and filename.endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    posts_data.append(df)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty or malformed file: {filename}")
        elif filename.startswith("reddit_comments_") and filename.endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    comments_data.append(df)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty or malformed file: {filename}")

    all_posts = pd.concat(posts_data, ignore_index=True) if posts_data else pd.DataFrame()
    all_comments = pd.concat(comments_data, ignore_index=True) if comments_data else pd.DataFrame()

    return all_posts, all_comments

if __name__ == "__main__":
    posts_df, comments_df = load_reddit_data("/home/ubuntu/upload")

    print("Posts DataFrame Head:")
    print(posts_df.head())
    print("\nComments DataFrame Head:")
    print(comments_df.head())

    print("\nPosts DataFrame Columns:")
    print(posts_df.columns)
    print("\nComments DataFrame Columns:")
    print(comments_df.columns)






# Next, we will proceed with data preprocessing to create prompt-response pairs.




import google.generativeai as genai

# --- LangChain and Gemini Setup ---
# To use Gemini with LangChain, you need to configure your Google API key.
# It's recommended to load this from an environment variable for security.
# Replace 'YOUR_API_KEY' with your actual Gemini API key.
# You can get an API key from Google AI Studio: https://aistudio.google.com/app/apikey

# Set your API key as an environment variable (e.g., GOOGLE_API_KEY)
# For example, in your terminal before running the script:
# export GOOGLE_API_KEY='YOUR_API_KEY'

# Or, you can directly set it in the script (less secure for production):
# os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

print("\nLangChain and Gemini setup code added. Remember to set your GOOGLE_API_KEY environment variable.")




from sklearn.model_selection import train_test_split
import re

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\[\]]', '', text)  # Remove special characters except common punctuation
        text = text.lower()  # Convert to lowercase
        text = text.strip()  # Remove leading/trailing whitespace
        return text
    return ""

def create_prompt_response_pairs(posts_df, comments_df):
    # Merge posts and comments on 'id' from posts and 'post_id' from comments
    # We want to link each post (prompt) to its comments (responses)
    merged_df = pd.merge(posts_df, comments_df, left_on='id', right_on='post_id', how='inner')

    # Combine title and selftext for the prompt
    # Fill NaN in 'selftext' with an empty string so it can be concatenated
    merged_df['full_prompt'] = merged_df['title'].fillna('') + " " + merged_df['selftext'].fillna('')
    merged_df['full_prompt'] = merged_df['full_prompt'].apply(clean_text)
    merged_df['comment_body'] = merged_df['comment_body'].apply(clean_text)

    # Filter out empty prompts or responses after cleaning
    merged_df = merged_df[merged_df['full_prompt'].str.strip() != '']
    merged_df = merged_df[merged_df['comment_body'].str.strip() != '']

    # For story generation, we can consider each post as a prompt and its comments as potential continuations/responses.
    # We'll create a dataset where each row is a (prompt, response) pair.
    # A post can have multiple comments, so we'll have multiple entries for a single post.
    prompt_response_data = merged_df[['full_prompt', 'comment_body']].rename(columns={'comment_body': 'response'})

    return prompt_response_data

def split_data(data_df, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    return train_df, val_df

if __name__ == '__main__':
    posts_df, comments_df = load_reddit_data('/home/ubuntu/upload')

    print('\n--- Data Preprocessing ---')
    prompt_response_pairs = create_prompt_response_pairs(posts_df, comments_df)
    print(f'Total prompt-response pairs: {len(prompt_response_pairs)}')
    print('Prompt-Response Pairs Head:')
    print(prompt_response_pairs.head())

    train_data, val_data = split_data(prompt_response_pairs)
    print(f'Training data size: {len(train_data)}')
    print(f'Validation data size: {len(val_data)}')

    # Save processed data for later use
    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
    print('\nProcessed data saved to train_data.csv and val_data.csv')




from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Fine-tuning Pipeline (Conceptual) ---
# Direct fine-tuning of Gemini models is typically done through Google Cloud Vertex AI.
# LangChain provides an excellent framework for interacting with LLMs, including Gemini,
# and for managing prompts and chains. While LangChain doesn't directly offer a 'fine-tune'
# function for Gemini models in the same way you might fine-tune a smaller open-source model,
# it can be used to prepare data for fine-tuning on Vertex AI, or to implement few-shot learning
# with your processed data.

# For the purpose of demonstrating the pipeline, we will show how to set up a LangChain
# interaction with a Gemini model for story generation, and how you would conceptually
# integrate your processed data.

# 1. Initialize the Gemini Model
# You can specify the model name, e.g., 'gemini-pro'.
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# 2. Define the Prompt Template
# This template will guide the model in generating short stories based on your prompts.
# We'll use the 'full_prompt' from your Reddit data as the input.
story_prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""You are a creative short story writer. Based on the following prompt, write a compelling short story:

Prompt: {prompt}

Story:"""
)

# 3. Create an LLM Chain
# This chain combines the prompt template with the Gemini LLM.
story_chain = LLMChain(llm=llm, prompt=story_prompt_template)

# --- How to use your processed data for 'training' or few-shot learning ---
# Since direct fine-tuning is via Vertex AI, here's how you'd use your data:

# Option A: Few-Shot Learning with LangChain (if your data is small enough)
# You can create examples from your train_data.csv and include them in your prompt.
# This is effective for smaller datasets or for guiding the model's style.
# Example (this would be done within a more complex prompt engineering setup):
# from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# from langchain.chains import LLMChain

# examples = []
# # Load a few examples from your train_data.csv
# # For demonstration, let's assume you have a list of dicts like:
# # examples = [
# #     {"prompt": "A lone astronaut discovers a strange artifact on Mars.", "response": "The artifact pulsed with an eerie light..."},
# #     # ... more examples
# # ]

# # example_prompt = PromptTemplate(input_variables=["prompt", "response"], template="Prompt: {prompt}\nStory: {response}")
# # few_shot_prompt = FewShotPromptTemplate(
# #     examples=examples,
# #     example_prompt=example_prompt,
# #     prefix="Write a short story based on the given prompt. Here are some examples:",
# #     suffix="\nPrompt: {prompt}\nStory:",
# #     input_variables=["prompt"],
# #     example_separator="\n\n"
# # )

# # few_shot_chain = LLMChain(llm=llm, prompt=few_shot_prompt)

# Option B: Preparing Data for Vertex AI Fine-tuning
# Your `train_data.csv` and `val_data.csv` are already in a suitable format (prompt, response pairs).
# You would typically upload these to Google Cloud Storage and then use Vertex AI's API or UI
# to initiate a fine-tuning job. The `full_prompt` column would be your input, and `response`
# would be your target output for the model to learn from.

# Example of how you would use the chain for inference after hypothetical fine-tuning or with few-shot:
if __name__ == '__main__':
    # ... (previous data loading and preprocessing code)

    # Example of generating a story using the defined chain
    sample_prompt = "A detective investigates a mysterious disappearance in a foggy, old town."
    print(f"\n--- Generating Story for Prompt: {sample_prompt} ---")
    # For actual generation, you would call:
    # generated_story = story_chain.run(prompt=sample_prompt)
    # print(generated_story)
    print("To generate a story, uncomment the 'story_chain.run' line and ensure GOOGLE_API_KEY is set.")
    print("Note: This is a conceptual demonstration. Actual fine-tuning of Gemini models is done via Google Cloud Vertex AI.")





# --- Model Testing and Validation (Conceptual) ---
# Since direct fine-tuning is handled by Vertex AI, the evaluation metrics
# for the fine-tuned model would typically be provided by Vertex AI itself.
# However, we can demonstrate how you would conceptually load a model (or use the base model)
# and perform inference for validation or testing purposes.

# For evaluating a generative model like Gemini for short story generation,
# traditional metrics (like accuracy) are not directly applicable.
# Instead, human evaluation or metrics like ROUGE, BLEU (for text similarity),
# or more advanced metrics that assess coherence, creativity, and relevance
# would be used. LangChain can help with setting up the inference pipeline.

# 1. Load the (hypothetically) fine-tuned model or use the base model
# In a real scenario with Vertex AI, you would deploy your fine-tuned model
# and then interact with its endpoint. Here, we continue with the base Gemini model.
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7) # Already initialized above

# 2. Generate Inferences on Validation Data
# We will take a few samples from the validation set and generate stories.
if __name__ == '__main__':
    # ... (previous data loading, preprocessing, and fine-tuning conceptual code)

    print("\n--- Model Testing and Validation ---")
    # Load validation data (assuming it was saved)
    try:
        val_data = pd.read_csv("val_data.csv")
    except FileNotFoundError:
        print("Validation data (val_data.csv) not found. Please run the data preprocessing step first.")
        val_data = pd.DataFrame()

    if not val_data.empty:
        print("Generating sample stories from validation prompts...")
        sample_val_prompts = val_data["full_prompt"].sample(min(5, len(val_data)), random_state=42).tolist()

        for i, prompt in enumerate(sample_val_prompts):
            print(f"\nSample {i+1} Prompt: {prompt}")
            # In a real scenario, you would call the model here:
            # generated_story = story_chain.run(prompt=prompt)
            # print(f"Generated Story: {generated_story}")
            print("Generated Story: [Story generation requires GOOGLE_API_KEY and actual model inference.]")

    print("\nConceptual code for model testing and validation added. Actual evaluation would involve human review or advanced NLP metrics.")


