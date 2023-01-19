import tweepy
import keys
import PCA_3d
from generate_similar_words import *
import re
from word_classifier import *
from text_generation import *

def api():
    auth = tweepy.OAuth1UserHandler(
        keys.api_key , keys.api_secret,
        keys.access_token, keys.access_token_secret
    )
    api = tweepy.API(auth)

    return api

def tweet(api: tweepy.API, message: str, image_path=None):
    api.update_status(message)

    print("Tweeted successfuly!")

if __name__ == '__main__':

    api = api()

    mention = api.mentions_timeline()[0]

    #PCA_3d.display_pca_scatterplot_2D(model, user_input, similar_word, labels, color_map)

    print(str(mention.id) + ' - ' + mention.text)
    #answer
    if '#hello' in mention.text.lower():
        print("found #hello")
        print("responding back...")
        #print(mention.id)

        #post on twitter

        status = '@' + mention.user.screen_name + " hello!"
        in_reply_to_status_id = mention.id
        api.update_status(status=status, in_reply_to_status_id=in_reply_to_status_id)


    elif '#embedding' in mention.text.lower():
        print("found #embedding")
        print("responding back...")
        text = mention.text
        input_word = [word for word in text.split() if word.startswith('$')][0][1:]
        print(input_word)

        user_input = [x.strip() for x in input_word.split(',')]
        result_word = []

        for words in user_input:
            sim_words = model.most_similar(words, topn=20)
            sim_words = append_list(sim_words, words)

            result_word.extend(sim_words)

        similar_word = [word[0] for word in result_word]
        similarity = [word[1] for word in result_word]
        similar_word.extend(user_input)
        labels = [word[2] for word in result_word]
        label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
        color_map = [label_dict[x] for x in labels]

        PCA_3d.display_pca_scatterplot_2D(model, user_input, similar_word, labels, color_map)

        #post on twitter

        status = '@' + mention.user.screen_name + " Connected words:"
        filename = "figure.png"
        in_reply_to_status_id = mention.id
        api.update_status_with_media(
            status=status,
            filename=filename,
            in_reply_to_status_id=in_reply_to_status_id
        )

    elif '#emotion_detection' in mention.text.lower():
        print("found #word_emotion")
        print("responding back...")

        sentence = mention.text
        sentence = " ".join(filter(lambda x:x[0]!='#', sentence.split()))
        sentence = " ".join(filter(lambda x: x[0] != '@', sentence.split()))
        print(sentence)

        sequence = get_sequences(tokenizer, [sentence, ])[0]
        print(sequence)

        p = model_emotions.predict(np.expand_dims(sequence, axis=0))
        classof = np.argmax(p, axis=1)

        print(index_to_classes.get(classof[0]))


        status = '@' + mention.user.screen_name + " Detected: " + index_to_classes.get(classof[0])
        in_reply_to_status_id = mention.id
        api.update_status(status=status, in_reply_to_status_id=in_reply_to_status_id)

    else:
        print("found nothing")
        print("responding back...")

        question = mention.text
        answer = generator(question, max_length = 30, num_return_sequences=1)[0]['generated_text']


        status = '@' + mention.user.screen_name + " " + answer
        in_reply_to_status_id = mention.id
        api.update_status(status=status, in_reply_to_status_id=in_reply_to_status_id)

