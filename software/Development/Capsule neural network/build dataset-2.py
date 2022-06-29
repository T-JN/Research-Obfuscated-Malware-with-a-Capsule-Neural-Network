
from scipy.spatial.distance import cosine

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read *.json Vectors
# *.json_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# *.json_vector_file = 'data/corpus/' + dataset + '_*.json_vectors.txt'
#_, embd, *.json_vector_map = load*.json2Vec(*.json_vector_file)
# *.json_embeddings_dim = len(embd[0])

*.json_embeddings_dim = 300
*.json_vector_map = {}

# shulffing
*.json_name_list = []
*.json_train_list = []
*.json_test_list = []

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    *.json_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        *.json_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        *.json_train_list.append(line.strip())
f.close()
# print(*.json_train_list)
# print(*.json_test_list)

*.json_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    *.json_content_list.append(line.strip())
f.close()
# print(*.json_content_list)

train_ids = []
for train_name in *.json_train_list:
    train_id = *.json_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in *.json_test_list:
    test_id = *.json_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
print(ids)
print(len(ids))

shuffle_*.json_name_list = []
shuffle_*.json_*.jsons_list = []
for id in ids:
    shuffle_*.json_name_list.append(*.json_name_list[int(id)])
    shuffle_*.json_*.jsons_list.append(*.json_content_list[int(id)])
shuffle_*.json_name_str = '\n'.join(shuffle_*.json_name_list)
shuffle_*.json_*.jsons_str = '\n'.join(shuffle_*.json_*.jsons_list)

f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_*.json_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_*.json_*.jsons_str)
f.close()

# build vocab
*.json_freq = {}
*.json_set = set()
for *.json_*.jsons in shuffle_*.json_*.jsons_list:
    *.jsons = *.json_*.jsons.split()
    for *.json in *.jsons:
        *.json_set.add(*.json)
        if *.json in *.json_freq:
            *.json_freq[*.json] += 1
        else:
            *.json_freq[*.json] = 1

vocab = list(*.json_set)
vocab_size = len(vocab)

*.json_*.json_list = {}

for i in range(len(shuffle_*.json_*.jsons_list)):
    *.json_*.jsons = shuffle_*.json_*.jsons_list[i]
    *.jsons = *.json_*.jsons.split()
    appeared = set()
    for *.json in *.jsons:
        if *.json in appeared:
            continue
        if *.json in *.json_*.json_list:
            *.json_list = *.json_*.json_list[*.json]
            *.json_list.append(i)
            *.json_*.json_list[*.json] = *.json_list
        else:
            *.json_*.json_list[*.json] = [i]
        appeared.add(*.json)

*.json_*.json_freq = {}
for *.json, *.json_list in *.json_*.json_list.items():
    *.json_*.json_freq[*.json] = len(*.json_list)

*.json_id_map = {}
for i in range(vocab_size):
    *.json_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
*.json definitions begin
'''
'''
definitions = []
for *.json in vocab:
    *.json = *.json.strip()
    synsets = wn.synsets(clean_str(*.json))
    *.json_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        *.json_defs.append(syn_def)
    *.json_des = ' '.join(*.json_defs)
    if *.json_des == '':
        *.json_des = '<PAD>'
    definitions.append(*.json_des)
string = '\n'.join(definitions)
f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()
tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))
*.json_vectors = []
for i in range(len(vocab)):
    *.json = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    *.json_vector = *.json + ' ' + temp
    *.json_vectors.append(*.json_vector)
string = '\n'.join(*.json_vectors)
f = open('data/corpus/' + dataset + '_*.json_vectors.txt', 'w')
f.write(string)
f.close()
*.json_vector_file = 'data/corpus/' + dataset + '_*.json_vectors.txt'
_, embd, *.json_vector_map = load*.json2Vec(*.json_vector_file)
*.json_embeddings_dim = len(embd[0])
'''

'''
*.json definitions end
'''

# label list
label_set = set()
for *.json_meta in shuffle_*.json_name_list:
    temp = *.json_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# x: feature vectors of training *.jsons, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_*.json_names = shuffle_*.json_name_list[:real_train_size]
real_train_*.json_names_str = '\n'.join(real_train_*.json_names)

f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_*.json_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    *.json_vec = np.array([0.0 for k in range(*.json_embeddings_dim)])
    *.json_*.jsons = shuffle_*.json_*.jsons_list[i]
    *.jsons = *.json_*.jsons.split()
    *.json_len = len(*.jsons)
    for *.json in *.jsons:
        if *.json in *.json_vector_map:
            *.json_vector = *.json_vector_map[*.json]
            # print(*.json_vec)
            # print(np.array(*.json_vector))
            *.json_vec = *.json_vec + np.array(*.json_vector)

    for j in range(*.json_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(*.json_vec[j] / *.json_len)  # *.json_vec[j]/ *.json_len

# x = sp.csr_matrix((real_train_size, *.json_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, *.json_embeddings_dim))

y = []
for i in range(real_train_size):
    *.json_meta = shuffle_*.json_name_list[i]
    temp = *.json_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(y)

# tx: feature vectors of test *.jsons, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    *.json_vec = np.array([0.0 for k in range(*.json_embeddings_dim)])
    *.json_*.jsons = shuffle_*.json_*.jsons_list[i + train_size]
    *.jsons = *.json_*.jsons.split()
    *.json_len = len(*.jsons)
    for *.json in *.jsons:
        if *.json in *.json_vector_map:
            *.json_vector = *.json_vector_map[*.json]
            *.json_vec = *.json_vec + np.array(*.json_vector)

    for j in range(*.json_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(*.json_vec[j] / *.json_len)  # *.json_vec[j] / *.json_len

# tx = sp.csr_matrix((test_size, *.json_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, *.json_embeddings_dim))

ty = []
for i in range(test_size):
    *.json_meta = shuffle_*.json_name_list[i + train_size]
    temp = *.json_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> *.jsons

*.json_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, *.json_embeddings_dim))

for i in range(len(vocab)):
    *.json = vocab[i]
    if *.json in *.json_vector_map:
        vector = *.json_vector_map[*.json]
        *.json_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    *.json_vec = np.array([0.0 for k in range(*.json_embeddings_dim)])
    *.json_*.jsons = shuffle_*.json_*.jsons_list[i]
    *.jsons = *.json_*.jsons.split()
    *.json_len = len(*.jsons)
    for *.json in *.jsons:
        if *.json in *.json_vector_map:
            *.json_vector = *.json_vector_map[*.json]
            *.json_vec = *.json_vec + np.array(*.json_vector)

    for j in range(*.json_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(*.json_vec[j] / *.json_len)  # *.json_vec[j]/*.json_len
for i in range(vocab_size):
    for j in range(*.json_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(*.json_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, *.json_embeddings_dim))

ally = []
for i in range(train_size):
    *.json_meta = shuffle_*.json_name_list[i]
    temp = *.json_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
*.json *.json heterogeneous graph
'''

# *.json co-occurence with context windows
window_size = 20
windows = []

for *.json_*.jsons in shuffle_*.json_*.jsons_list:
    *.jsons = *.json_*.jsons.split()
    length = len(*.jsons)
    if length <= window_size:
        windows.append(*.jsons)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = *.jsons[j: j + window_size]
            windows.append(window)
            # print(window)


*.json_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in *.json_window_freq:
            *.json_window_freq[window[i]] += 1
        else:
            *.json_window_freq[window[i]] = 1
        appeared.add(window[i])

*.json_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            *.json_i = window[i]
            *.json_i_id = *.json_id_map[*.json_i]
            *.json_j = window[j]
            *.json_j_id = *.json_id_map[*.json_j]
            if *.json_i_id == *.json_j_id:
                continue
            *.json_pair_str = str(*.json_i_id) + ',' + str(*.json_j_id)
            if *.json_pair_str in *.json_pair_count:
                *.json_pair_count[*.json_pair_str] += 1
            else:
                *.json_pair_count[*.json_pair_str] = 1
            # two orders
            *.json_pair_str = str(*.json_j_id) + ',' + str(*.json_i_id)
            if *.json_pair_str in *.json_pair_count:
                *.json_pair_count[*.json_pair_str] += 1
            else:
                *.json_pair_count[*.json_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

for key in *.json_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = *.json_pair_count[key]
    *.json_freq_i = *.json_window_freq[vocab[i]]
    *.json_freq_j = *.json_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * *.json_freq_i * *.json_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# *.json vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in *.json_vector_map and vocab[j] in *.json_vector_map:
            vector_i = np.array(*.json_vector_map[vocab[i]])
            vector_j = np.array(*.json_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# *.json *.json frequency
*.json_*.json_freq = {}

for *.json_id in range(len(shuffle_*.json_*.jsons_list)):
    *.json_*.jsons = shuffle_*.json_*.jsons_list[*.json_id]
    *.jsons = *.json_*.jsons.split()
    for *.json in *.jsons:
        *.json_id = *.json_id_map[*.json]
        *.json_*.json_str = str(*.json_id) + ',' + str(*.json_id)
        if *.json_*.json_str in *.json_*.json_freq:
            *.json_*.json_freq[*.json_*.json_str] += 1
        else:
            *.json_*.json_freq[*.json_*.json_str] = 1

for i in range(len(shuffle_*.json_*.jsons_list)):
    *.json_*.jsons = shuffle_*.json_*.jsons_list[i]
    *.jsons = *.json_*.jsons.split()
    *.json_*.json_set = set()
    for *.json in *.jsons:
        if *.json in *.json_*.json_set:
            continue
        j = *.json_id_map[*.json]
        key = str(i) + ',' + str(j)
        freq = *.json_*.json_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_*.json_*.jsons_list) /
                  *.json_*.json_freq[vocab[j]])
        weight.append(freq * idf)
        *.json_*.json_set.add(*.json)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

