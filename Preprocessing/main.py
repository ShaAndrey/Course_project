import os


def collect_data(folder, out_name):
    path = os.path.join(os.getcwd(), folder)
    with open(os.path.join(os.getcwd(), out_name), 'w') as out:
        for filename in os.listdir(path):
            with open(os.path.abspath(os.path.join(path, filename)), 'r') as file:
                for paragraph in file.read().splitlines():
                    out.write(paragraph.replace('<br />', '').lower().rstrip() + ' ')
                out.write('\n')


collect_data('../aclImdb/train/pos', 'X_train_pos')
collect_data('../aclImdb/train/neg', 'X_train_neg')
collect_data('../aclImdb/test/pos', 'X_test_pos')
collect_data('../aclImdb/test/neg', 'X_test_neg')
# collect_data('Waste', 'waste')
