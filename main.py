import lstm
import word2vec
import DataProcessor


def main():
    # learn embeddings
    word2vec.word2vec()
    # convert training,test and eval data into np arrays
    DataProcessor.build_data()
    # this calculates sentiments for the data
    lstm.lstm_script()


if __name__ == '__main__':
    main()
