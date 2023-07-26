import re
import lap
import time
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def gener_align_list_dict():
    #The aligned entity pairs generating.
    zh_en_file = open("../datasets/zh_en/ent_ILLs", 'r', encoding="UTF-8")
    patten_zh = "http://zh.dbpedia.org/resource/(.*?)\t"
    patten_en = "http://dbpedia.org/resource/(.*?)\n"
    contents = zh_en_file.readlines()
    align_list = [[re.findall(patten_zh, content)[0].replace("_", " ").lower(), re.findall(patten_en, content)[0].replace("_", " ").lower()]
                  for content in contents]
    #The translation dictionary generating.
    file_zh_trans = open("../translate files/zh_en/zh_align_trans.txt", 'r', encoding="UTF-8")
    zh_trans_dict = {}
    for content in file_zh_trans.readlines():
        content_list = content.replace("\n", "").split("\t")
        zh_trans_dict[content_list[0].lower()] = content_list[1].lower()
    return align_list, zh_trans_dict


def character_encode(align_list, trans_dict, encode_model):
    model = SentenceTransformer("sentence-transformers/"+encode_model)
    print("the encoding of aligned entity name is starting...", end="\t\t\t")
    zh_encode, en_encode = model.encode([trans_dict[content[0]] for content in align_list]), model.encode([content[1] for content in align_list])
    print("encoding is end.")
    encode_result = [[zh_encode[index], en_encode[index]] for index in range(len(align_list))]
    return encode_result


def gener_sim_mat(encode_vec_list):
    vec_rep_matrix_l = np.array([vec_list[0] for vec_list in encode_vec_list]).astype("float")
    vec_rep_matrix_r = np.array([vec_list[1] for vec_list in encode_vec_list]).astype("float")
    norm_l = np.array([np.linalg.norm(vec_list[0]) for vec_list in encode_vec_list]).reshape(-1, 1)
    norm_r = np.array([np.linalg.norm(vec_list[1]) for vec_list in encode_vec_list]).reshape(-1, 1)
    norm = np.dot(norm_l, norm_r.T)
    norm = np.where(norm==0, float("inf"), norm)
    sim_mat = (np.dot(vec_rep_matrix_l, vec_rep_matrix_r.T)/norm)*0.5+0.5
    return sim_mat


def get_info_struct_att(align_list, choice):
    #The structure and attribute information generating.
    #"choice": str_tar(the tail entity of structure triples)/att_rel(the relation of attribute triples)/att_tar(the tail entity of attribute triples)
    def get_att_triples():
        path_zh = "../../datasets/zh_en/zh_att_triples"
        path_en = "../../datasets/zh_en/en_att_triples"
        file_zh = open(path_zh, 'r', encoding="UTF-8")
        file_en = open(path_en, 'r', encoding="UTF-8")
        triples_zh = []
        triples_en = []
        for content in file_zh.readlines():
            content_list = content.split("> ")
            triples_zh.append([content_list[0].split("/")[-1].replace("_", " "), content_list[1].split("/")[-1].replace("_", " "),
                               re.findall(r"\"(.*?)\"", content_list[2]+"\"\"")[0].replace("_", " ")])
        for content in file_en.readlines():
            content_list = content.split("> ")
            triples_en.append([content_list[0].split("/")[-1].replace("_", " "), content_list[1].split("/")[-1].replace("_", " "),
                               re.findall(r"\"(.*?)\"", content_list[2]+"\"\"")[0].replace("_", " ")])
        return triples_zh, triples_en

    def get_str_triples():
        path_zh = "../datasets/zh_en/zh_rel_triples"
        path_en = "../datasets/zh_en/en_rel_triples"
        file_zh = open(path_zh, 'r', encoding="UTF-8")
        file_en = open(path_en, 'r', encoding="UTF-8")
        triples_zh = []
        triples_en = []
        for content in file_zh.readlines():
            content_list = content.split("\t")
            triples_zh.append([content_list[0].split("/")[-1].replace("_", " "),
                               content_list[1].split("/")[-1].replace("_", " "),
                               content_list[2].split("/")[-1].replace("\n", "").replace("_", " ")])
        for content in file_en.readlines():
            content_list = content.split("\t")
            triples_en.append([content_list[0].split("/")[-1].replace("_", " "),
                               content_list[1].split("/")[-1].replace("_", " "),
                               content_list[2].split("/")[-1].replace("\n", "").replace("_", " ")])
        return triples_zh, triples_en

    def str_handle(string):
        #The special characters deleting and the strings splitting.
        if len(string)==0:
            return string
        temp_string = string[0]+"".join([(" " + string[index]) if string[index-1].islower() and string[index].isupper()  else string[index]
                      if string[index].isalpha() else " " for index in range(1, len(list(string)), 1)]).replace("of ", " ").lower()
        new_string = " ".join([word if len(word)>1 else "none" for word in temp_string.split(" ")]).replace(" none", "").lower()
        return new_string

    def get_str_att_info(type, choice):
        #The structure and attribute information generating.
        #param "type": "str" or "att"
        #param "choice": "rel" or "tar"
        if type=="str":
            triples_zh, triples_en = get_str_triples()
        if type=="att":
            triples_zh, triples_en = get_att_triples()
        value = 0
        if choice=="rel":
            value = 1
        if choice=="tar":
            value = 2
        zh_info = []
        en_info = []
        for index in tqdm(range(len(align_list))):
            zh_temp = [align_list[index][0]]
            for triple_zh in triples_zh:
                if align_list[index][0]==triple_zh[0]:
                    zh_temp.append(str_handle(triple_zh[value]))
                zh_info.append(zh_temp)
            en_temp = [align_list[index][1]]
            for triple_en in triples_en:
                if align_list[index][1].replace(" ", "")==triple_en[0].replace(" ", "").lower():
                    en_temp.append(str_handle(triple_en[value]))
            en_info.append(en_temp)
        return zh_info, en_info

    if choice=="att_rel":
        zh_att_rel, en_att_rel=get_str_att_info("att", "rel")
        np.save("../multi-view infomation/zh_en/zh_att_rel.npy", zh_att_rel)
        np.save("../multi-view infomation/zh_en/en_att_rel.npy", en_att_rel)
        return zh_att_rel, en_att_rel

    if choice=="att_tar":
        zh_att_tar, en_att_tar=get_str_att_info("att", "tar")
        np.save("../multi-view infomation/zh_en/zh_att_tar.npy", zh_att_tar)
        np.save("../multi-view infomation/zh_en/en_att_tar.npy", en_att_tar)
        return zh_att_tar, en_att_tar

    if choice=="str_tar":
        zh_str_tar, en_str_tar=get_str_att_info("str", "tar")
        np.save("../multi-view infomation/zh_en/zh_str_tar.npy", zh_str_tar)
        np.save("../multi-view infomation/zh_en/en_str_tar.npy", en_str_tar)
        return zh_str_tar, en_str_tar


def gener_att_str_info_vector(list_info, file_path, type, encode_model):
    #The final vector representation generating.
    def is_exist_non_en(string):
        #To judge a string whether existing non-English characters.
        for char in string.replace(" ", "").lower():
            if 97<=ord(char)<=122:
                continue
            return True

    def encode_vector(vector_dict, word_list):
        #Using the summation method to generate the final vector representation of extra information.
        vector_word_list = [vector_dict[word] for word in word_list]
        sum = np.zeros(768) #The vector dimension is 768 which is generated by the encoder language model.
        for vector_word in vector_word_list:
            sum += vector_word
        average_sum = sum/(len(vector_word_list)+1)
        '''
        #The maxima or minima method to generate final vector.
        vector = np.zeros(768)
        for vector_word in vector_word_list:
            if np.linalg.norm(vector)<np.linalg.norm(vector_word): #minima encoding mode 
                vector = vector_word
            if np.linalg.norm(vector)>np.linalg.norm(vector_word): #maxima encoding mode
                vector = vector_word
        '''
        return average_sum

    if type in ["zh", "ja", "fr"]:
        file_trans = open(file_path, 'r', encoding="UTF-8")
        trans_dict = {content.split("\t")[0]:content.split("\t")[1].replace("\n", "").lower() for content in file_trans.readlines()}
        summary_list = list(set([trans_dict[index_2] if is_exist_non_en(index_2) else index_2 for index_1 in list_info for index_2 in index_1[1::]]))
        model = SentenceTransformer("sentence-transformers/"+encode_model)
        print("the encoding of relevant information is starting...(zh)", end="\t\t\t")
        encode_list = model.encode(summary_list)
        print("encoding is end.")
        summary_dict = {summary_list[index]:encode_list[index] for index in range(len(summary_list))}
        result_dict = {}
        for content in list_info:
            word_list = [trans_dict[word] if is_exist_non_en(word) else word for word in content[1::]]
            result_dict[content[0]] = encode_vector(summary_dict, word_list)
    else:
        summary_list = list(set([index_2 for index_1 in list_info for index_2 in index_1[1::]]))
        model = SentenceTransformer("sentence-transformers/"+encode_model)
        print("the encoding of relevant information is starting...(en)", end="\t\t\t")
        encode_list = model.encode(summary_list)
        print("encoding is end.")
        summary_dict = {summary_list[index]:encode_list[index] for index in range(len(summary_list))}
        result_dict = {content[0]:encode_vector(summary_dict, content[1::]) for content in list_info}
    return result_dict


if __name__=="__main__":
    start_time = time.time()
    align_list, zh_trans_dict = gener_align_list_dict() #To attain the aligned entity pairs and translation dictionary of no-English language.

    model = "msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch" #Encoded by multi-language Sentence-Bert.
    print("entity name:")
    sim_mat = gener_sim_mat(character_encode(align_list, zh_trans_dict, model)) #To attain the similarity matrix of aligned entity pairs.

    #To attain the similarity matrix of structure triples tail entity, attribute triples relation, and attribute triples tail entity respectively.
    for infor_name in ["str_tar", "att_rel", "att_tar"]:
        print(infor_name+":")
        zh_vec_dict = gener_att_str_info_vector(np.load("../multi-view infomation/zh_en/zh_"+infor_name+".npy", allow_pickle=True),
                                                "../translate files/zh_en/zh_"+infor_name+"_trans.txt", "zh", model)
        en_vec_dict = gener_att_str_info_vector(np.load("../multi-view infomation/zh_en/en_"+infor_name+".npy", allow_pickle=True), "", "en", model)
        encode_vec_list = [[zh_vec_dict[align[0]], en_vec_dict[align[1]]] for align in align_list]
        if infor_name=="att_rel":
            sim_mat += gener_sim_mat(encode_vec_list)*0.15
        else:
            sim_mat += gener_sim_mat(encode_vec_list)*0.75

    sim_mat += sim_mat.T
    list_x = [index for index in range(len(sim_mat))]
    _, list_y, _ = lap.lapjv(sim_mat)
    sim_mat_ra = sim_mat[:, list_y]
    sim_mat_ra -= np.array([sim_mat_ra[index][index] for index in range(sim_mat_ra.shape[0])])[:, None]
    sim_mat_ra += sim_mat_ra.T

    #To calculate the values of Hits@1 and Hits@10.
    copy_sim_mat_ra = sim_mat_ra.copy()
    copy_sim_mat_ra -= copy_sim_mat_ra.mean(axis=0)[:, None]
    sim_mat_ra_temp = np.where(copy_sim_mat_ra>0, copy_sim_mat_ra, 0)
    sort_sim_mat_ra = sim_mat_ra_temp.argsort(axis=1)
    for hits in [1, 10]:
        align_count = 0
        hits_mat = np.array(sort_sim_mat_ra[:, -hits::]).T
        for index in range(hits):
            new_rank = np.array([list_y[key] for key in hits_mat[index, :]])
            align_count += np.sum((list_x==new_rank)!=0)
        print("the Hits@%d of entity alignment is %.4f" %(hits, align_count/len(sim_mat)))

    #To calculate the value of Mean Reciprocal Rank.
    mrr = 0
    sort_sim_mat_ra = sim_mat_ra.argsort(axis=1).T
    for index in tqdm(range(len(sim_mat))):
        new_rank = np.array([list_y[key] for key in sort_sim_mat_ra[len(sim_mat)-(index+1), :]])
        mrr += np.sum(1/np.where(((list_x==new_rank)!=0)>0, (index+1), float("inf")))/len(sim_mat)
    print("the value of MRR is %.4f" %mrr)
    print("this program is spending time: %.2fs" %(time.time()-start_time))
