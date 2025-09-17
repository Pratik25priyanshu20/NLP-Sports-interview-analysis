#src/text_generation.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import re

class TextGenerator:
    def __init__(self, model_name="gpt2", models_dir="models", results_dir="results"):
        """Initialize TextGenerator with model name and directories."""
        self.model_name = model_name
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(self.results_dir, "text_generation"), exist_ok=True)
        
        print(f"Initializing text generator with {model_name}...")
        
        # Load tokenizer and model for generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # If the tokenizer doesn't have a padding token, set it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create text generation pipeline
        self.text_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("Text generator initialized successfully.")
    
    def load_category_examples(self, data_path):
        """Load examples of each interview category for prompt engineering."""
        print("\nLoading category examples...")
        
        # Load processed training data
        train_df = pd.read_csv(data_path)
        
        # Group by label and get examples
        self.category_examples = {}
        for label in train_df['Labels'].unique():
            category_df = train_df[train_df['Labels'] == label]
            # Get 3 random examples
            examples = category_df.sample(min(3, len(category_df)))
            self.category_examples[label] = examples['Interview Text'].tolist()
        
        print(f"Loaded examples for {len(self.category_examples)} categories.")
        
        return self.category_examples
    
    def generate_response(self, category, question, max_length=250, num_return_sequences=1):
        """Generate a realistic interview response based on category and question."""
        print(f"\nGenerating response for category: {category}, question: {question}")
        
        # Create prompt with category examples and question
        examples_text = ""
        if category in self.category_examples:
            examples = self.category_examples[category]
            for i, example in enumerate(examples[:2]):  # Use up to 2 examples to avoid token limit
                examples_text += f"Example {i+1}:\n{example}\n\n"
        
        prompt = f"""
The following is an interview in the category: {category}

{examples_text}
Question: {question}

Response:"""
        
        # Generate text
        generated_texts = self.text_generator(
            prompt,
            max_length=len(prompt.split()) + max_length,
            num_return_sequences=num_return_sequences,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Extract response (removing the prompt)
        responses = []
        for item in generated_texts:
            response = item['generated_text'][len(prompt):]
            # Clean up response (remove incomplete sentences at the end)
            response = re.sub(r'[^.!?]*$', '', response.strip())
            responses.append(response)
        
        return responses[0] if num_return_sequences == 1 else responses
    
    def generate_example_responses(self, save=True):
        """Generate example responses for each category."""
        print("\nGenerating example responses for each category...")
        
        example_questions = {
            "Game Strategy": [
                "How did you adjust your defensive strategy in the second half?",
                "What formation do you plan to use against the upcoming opponent?"
            ],
            "Player Performance": [
                "How would you rate your performance in today's game?",
                "What did you think of the rookie's debut performance?"
            ],
            "Injury Updates": [
                "Can you give us an update on Johnson's knee injury?",
                "When do you expect to return to full training?"
            ],
            "Post-Game Analysis": [
                "What were the key moments that decided this match?",
                "How do you feel about the team's overall performance today?"
            ],
            "Team Morale": [
                "How is the team handling the pressure of the playoff race?",
                "What's the atmosphere like in the locker room after this losing streak?"
            ],
            "Upcoming Matches": [
                "How are you preparing for the championship match next week?",
                "What challenges do you expect from your next opponent?"
            ],
            "Off-Game Matters": [
                "How has your charity work influenced your perspective on the game?",
                "What are your thoughts on the recent trade rumors?"
            ],
            "Controversies": [
                "What's your response to the referee's controversial call in the final minutes?",
                "How do you feel about the league's new policy changes?"
            ]
        }
        
        # Generate responses
        all_examples = []
        for category, questions in example_questions.items():
            for question in questions[:1]:  # Use only one question per category to save time
                response = self.generate_response(category, question)
                example = {
                    "Category": category,
                    "Question": question,
                    "Response": response
                }
                all_examples.append(example)
        
        # Convert to DataFrame
        examples_df = pd.DataFrame(all_examples)
        
        # Save examples
        if save:
            examples_df.to_csv(os.path.join(self.results_dir, "text_generation", "example_responses.csv"), index=False)
            
            # Also save as markdown for better readability
            with open(os.path.join(self.results_dir, "text_generation", "example_responses.md"), 'w') as f:
                f.write("# Example Generated Interview Responses\n\n")
                for _, row in examples_df.iterrows():
                    f.write(f"## Category: {row['Category']}\n\n")
                    f.write(f"**Question:** {row['Question']}\n\n")
                    f.write(f"**Response:** {row['Response']}\n\n")
                    f.write("---\n\n")
            
            print(f"Example responses saved to {os.path.join(self.results_dir, 'text_generation')}")
        
        return examples_df
    
    def write_ethical_reflection(self):
        """Write a reflection on ethical implications of AI-generated sports interviews."""
        print("\nWriting ethical reflection...")
        
        reflection = """
# Ethical Implications of AI-Generated Sports Interview Content

## Potential Benefits
- **Content Creation Efficiency**: AI can help journalists and content creators quickly generate draft responses for common questions, saving time in content production.
- **Language Learning**: For international athletes, AI could help formulate more articulate responses in languages they're less comfortable with.
- **Consistency**: Teams could ensure messaging stays on-brand and consistent across multiple interviews.
- **Preparation**: Athletes and coaches could use AI-generated responses as preparation tools before actual interviews.

## Ethical Concerns
- **Authenticity and Trust**: Sports journalism relies on authentic human experiences and emotions. AI-generated content risks undermining the genuineness that fans value in athlete interviews.
- **Misinformation Risk**: If AI generates plausible but factually incorrect statements about injuries, game strategies, or team dynamics, this could spread misinformation.
- **Voice Appropriation**: Using AI to simulate a specific athlete's speaking style raises questions about personality rights and consent.
- **Media Credibility**: If audiences discover that interview content is AI-generated, it could further erode trust in sports journalism and media.
- **Employment Impact**: Widespread adoption could potentially reduce opportunities for sports journalists, particularly those covering minor leagues or less popular sports.

## Responsible Implementation Guidelines
1. **Transparency**: Any AI-generated content should be clearly labeled as such.
2. **Human Oversight**: AI should be used as an assistive tool with human editors reviewing and approving content before publication.
3. **Consent**: Athletes should provide explicit permission before their interview style is modeled or simulated.
4. **Fact Checking**: AI-generated responses should be verified for factual accuracy, especially regarding sensitive topics like injuries or team strategies.
5. **Complementary Use**: AI should complement rather than replace authentic athlete interviews, perhaps being used for routine questions while preserving human interaction for more nuanced topics.

## Conclusion
AI-generated sports interview content presents both opportunities and challenges for the industry. While it can enhance efficiency and provide useful tools for preparation, the core value of sports journalism lies in authentic human connection and storytelling. The technology should therefore be implemented thoughtfully, with clear ethical guidelines and a commitment to preserving the authenticity that makes sports compelling to fans.
"""
        
        # Save reflection
        with open(os.path.join(self.results_dir, "text_generation", "ethical_reflection.md"), 'w') as f:
            f.write(reflection)
        
        print(f"Ethical reflection saved to {os.path.join(self.results_dir, 'text_generation', 'ethical_reflection.md')}")
        
        return reflection


if __name__ == "__main__":
    # Initialize text generator
    generator = TextGenerator()
    
    # Load category examples
    generator.load_category_examples("data/processed/train_processed.csv")
    
    # Generate example responses
    generator.generate_example_responses()
    
    # Write ethical reflection
    generator.write_ethical_reflection()