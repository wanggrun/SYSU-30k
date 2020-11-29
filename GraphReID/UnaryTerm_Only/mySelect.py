import numpy as np
def mySelect(scores,labels):
    # scores: 80*1267ids, where 1267 is dims
    # labels: 80 labels, 80 = 10ids*8pic/id
    nums = scores.shape[0]  # 80
    dims = scores.shape[1]  # 1267
    # numpy to list
    labels_list = labels.tolist()
    # unique ids
    id_set = set(labels_list)
    # set to list
    id_list = list(id_set)
    id_num = len(id_list)
    imgs_num_per_id = labels.shape[0]//id_num                 ###
    labels_new = labels

    # print('labels_list:',id_list)
    # print('id_list:',id_list)
    scores = scores[:,id_list]      ##**
    ##############################################################
    for i in range(id_num):
        id_has_cards = np.zeros((id_num,1)) #local,0~9, actually, e.g.,2,33,56,77,...,1265
        # id_has_cards_ind = np.zeros((id_num,nums)) 
        id_has_cards_scores=[[] for i in range(id_num)]
        id_has_cards_ind=[[] for i in range(id_num)]
        for j in range(nums):
            one_card = scores[j,:] 
            higher_scores_ind = np.argmax(one_card,axis = 0)  ### exclude neg scores
            higher_scores = np.max(one_card,axis = 0)  ### exclude neg scores
            if higher_scores < -10000.0:
                  continue
            # id_local_ind = id_list.index(higher_scores_ind[0])  ##*
            id_local_ind = higher_scores_ind
            id_has_cards_ind[id_local_ind].append(j)
            id_has_cards_scores[id_local_ind].append(higher_scores)
            id_has_cards[id_local_ind,0] = id_has_cards[id_local_ind,0] + 1
        # find the biggest one
        biggest_one_ind = np.argmax(id_has_cards,axis = 0)  ###
        # step 1: set something, set all dims -1.0 scores if it is topk, = discard this card
        # if not topk, set one dims = 0.0
        attention_cards = id_has_cards_ind[biggest_one_ind[0]]
        attention_cards_scores = id_has_cards_scores[biggest_one_ind[0]]
        attention_cards_scores_np = np.array(attention_cards_scores)

        # sort:
        attention_cards_order_ind = np.argsort(-attention_cards_scores_np)
        # print('****************************************************************')
        # print('scores:',scores)
        # print('id_has_cards:',id_has_cards)
        # print('id_has_cards_scores:',id_has_cards_scores)
        # print('id_has_cards_ind:',id_has_cards_ind)
        # print('attention_cards_scores_np:',attention_cards_scores_np)
        # print('attention_cards_order_ind:',attention_cards_order_ind)

        assert(attention_cards_order_ind.shape[0]>=imgs_num_per_id)
        # for j in range(attention_cards_order_ind.shape[0]):
        for j in range(imgs_num_per_id):
            bigman_one_card = attention_cards[attention_cards_order_ind[j]]
            scores[bigman_one_card,:] = -100000.0
            scores[:,biggest_one_ind[0]] = -100000.0 
        # step 2: update labels
        attention_cards = np.array(attention_cards)
        labels_new[attention_cards] = id_list[biggest_one_ind[0]]    # > topk will be covered    #**  
    return labels_new   

if __name__ == '__main__':
    scores = np.random.rand(6,7)
    labels = np.array([1,0,1,2,0,2])
    labels_new = mySelect(scores,labels)
    print('labels_new:',labels_new)
