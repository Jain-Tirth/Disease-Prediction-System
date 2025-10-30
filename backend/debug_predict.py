import joblib
import json
import traceback

MODEL_PATH = 'backend/model.joblib'
FEATURES_PATH = 'backend/feature_names.json'

def main():
    try:
        print('Loading feature names...')
        with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
            features = json.load(f)
        print('Feature count:', len(features))

        print('Loading model...')
        model = joblib.load(MODEL_PATH)
        print('Model loaded:', type(model))

        # test vector of zeros
        x = [0] * len(features)
        print('Running predict on zero vector...')
        pred = model.predict([x])
        print('Prediction result:', pred)
    except Exception:
        print('Exception during debug prediction:')
        traceback.print_exc()

if __name__ == '__main__':
    main()
