import nltk
import re
import pandas as pd  # Data manipulation, processing and visualization
from nltk.stem import PorterStemmer  # Lemmatization, Stemming
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer  #converts sentences to vectors
from sklearn.linear_model import LogisticRegression
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

nltk.download('stopwords')

# Read train and test datasets
train_news = pd.read_csv('train.csv')
test_news = pd.read_csv('test.csv')

# Check for null values
train_news.isnull().sum()
test_news.isnull().sum()


# Clean news function for cleaning the content
def clean_news(news):
    ps = PorterStemmer()
    cleaned_content = re.sub('[^a-zA-Z]', " ", news)  # Remove non-alphabet characters
    cleaned_content = cleaned_content.lower()  # Convert to lowercase
    cleaned_content = cleaned_content.split()  # Split into words
    cleaned_content = [ps.stem(w) for w in cleaned_content if w not in stopwords.words('english')]  # Stem and remove stopwords
    cleaned_content = ' '.join(cleaned_content)  # Join words back into a string
    return cleaned_content


# Apply clean_news function to the datasets
train_news['news'] = train_news['news'].apply(clean_news)
test_news['news'] = test_news['news'].apply(clean_news)

# TF-IDF Vectorization
Tdf = TfidfVectorizer()
x_train = Tdf.fit_transform(train_news['news'].values)  # Fit and transform on training data
x_test = Tdf.transform(test_news['news'].values)  # Only transform on test data

# Logistic Regression Model to build a prediction model
model = LogisticRegression()
y_train = train_news['label'].values  # Training labels
y_test = test_news['label'].values  # Test labels

# Train the model on the training data
model.fit(x_train, y_train)

# Predict and evaluate on training data
train_prediction = model.predict(x_train)
print("Train accuracy:", accuracy_score(train_prediction, y_train))

# Predict and evaluate on test data
test_prediction = model.predict(x_test)
print("Test accuracy:", accuracy_score(test_prediction, y_test))


# Kivy App Class
class FakeNewsApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Input label and text input field
        self.input_label = Label(text="Enter news text:")
        self.layout.add_widget(self.input_label)

        self.news_input = TextInput(multiline=True)
        self.layout.add_widget(self.news_input)

        # Prediction button
        self.predict_button = Button(text="Predict Fake/Real")
        self.predict_button.bind(on_press=self.predict_news)
        self.layout.add_widget(self.predict_button)

        # Result label
        self.result_label = Label(text="Prediction will appear here")
        self.layout.add_widget(self.result_label)

        return self.layout

    def predict_news(self, instance):
        news_text = self.news_input.text
        cleaned_news = clean_news(news_text)  # Clean the input text

        # Transform the input text using the trained TF-IDF vectorizer
        input_vector = Tdf.transform([cleaned_news])

        # Predict using the trained model
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            self.result_label.text = "Fake News"
        else:
            self.result_label.text = "Real News"


# Run the Kivy App
if __name__ == '__main__':
    FakeNewsApp().run()
