import os
import torch
import gpt_handmade


class WordOperator:
    def read_text(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text


    def make_token(self, text):
        chars = sorted(list(set(text)))
        char_size = len(chars)
        char2int = { ch : i for i, ch in enumerate(chars) }
        int2char = { i : ch for i, ch in enumerate(chars) }
        self.encode = lambda a: [char2int[b] for b in a ]
        self.decode = lambda a: ''.join([int2char[b] for b in a ])

        train_data = torch.tensor(self.encode(text), dtype=torch.long)

        print("学習データで使っている文字数　：　", char_size)
        print("トークン化した学習データ　：　", train_data[:30])
        return train_data, char_size


def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    word_op = WordOperator()
    text = word_op.read_text('fewshotlearning.txt')
    train_data, char_size = word_op.make_token(text)
    
    # モデルの定義
    number_of_heads = 4 # 同時に実行されるself-attentionの数
    block_size = 8 # 一度に処理できる最大の文字数
    n_mbed = 32 # トークンの埋め込むベクトルの次元数
    batch_size = 32 # 同時に処理できる配列の数
    # char_size = len(train_data)

    model = gpt_handmade.ModelTransformer(n_mbed, char_size, block_size, number_of_heads, device)
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr =1e-3)

    for steps in range(10000):
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i : i + block_size] for i in  ix]).to(device)
        y = torch.stack([train_data[i+1 : i + block_size+1] for i in  ix]).to(device)
        logits, loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if steps % 100 == 0:
            print(f"step{steps} loss{loss.item()}")
    
    model.cpu()
    torch.save(model.state_dict(), 'model.pth')

def predict():
    device = 'cpu'

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    word_op = WordOperator()
    text = word_op.read_text('fewshotlearning.txt')
    train_data, char_size = word_op.make_token(text)
    idx = torch.zeros((1,1), dtype = torch.long)
    # モデルの定義
    number_of_heads = 4 # 同時に実行されるself-attentionの数
    block_size = 8 # 一度に処理できる最大の文字数
    n_mbed = 32 # トークンの埋め込むベクトルの次元数
    batch_size = 32 # 同時に処理できる配列の数
    # char_size = len(train_data)

    model = gpt_handmade.ModelTransformer(n_mbed, char_size, block_size, number_of_heads, device)
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
    with torch.no_grad():
        for _ in range(50):
            idx_pred = idx[:, -block_size:]
            logits , loss = model(idx_pred)
            logits = logits[:,-1,:]
            probs = torch.nn.functional.softmax(logits, dim=1)
            idx_next_pred = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next_pred),dim = 1)
        predict = word_op.decode(idx[0].tolist())
        print("予測結果 : ", predict)

if __name__ == '__main__':
    # train_model()
    predict()
    string = "\u5bb6"
    print(string)