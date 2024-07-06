from tkinter import *
from tkinter import filedialog
import os
import numpy as np
from nltk import sent_tokenize
import nltk
from string import punctuation
from nltk.corpus import stopwords

if True:
    root = Tk()
    my_filetypes = [('all files', '.*'), ('text files', '.txt')]
    root.filename =  filedialog.askopenfilename(title="Please select a file:",filetypes=my_filetypes)
    root.withdraw()
    try:
        text=open(root.filename, encoding="utf-8").read()
    except:
        text=open(root.filename).read()
    try:
        sentences=(sent_tokenize(text))
        emptyarray= np.empty((len(sentences),1,3),dtype=object)
        for s in range(len(sentences)):
            emptyarray[s][0][0] = sentences[s]
            emptyarray[s][0][1] = s
        stop_words = set(stopwords.words('english'))
        bi_token=[]
        bi_token_length=[]
        tri_token_length=[]
        for u in range(len(sentences)):
            sent_split1=[w.lower() for w in sentences[u].split(" ")]
            sent_split=[w for w in sent_split1 if w not in stop_words and w not in punctuation and not w.isdigit()]
            
            bigrams_list = [bigram for bigram in nltk.bigrams(sent_split)]
            bi_token.append(bigrams_list)
            bi_token_length.append(len(bi_token[u]))
        bi_tokens = [(int(o) / max(bi_token_length))*100 for o in bi_token_length]


        tri_token=[]
        for u in range(len(sentences)):
            sent_split2=[w.lower() for w in sentences[u].split(" ")]
            sent_split3=[w for w in sent_split2 if w not in stop_words and w not in punctuation and not w.isdigit()]
            trigrams_list = [trigram for trigram in nltk.trigrams(sent_split3)]
            tri_token.append(trigrams_list)
            tri_token_length.append(len(tri_token[u]))
        tri_tokens = [(int(m) / max(tri_token_length))*100 for m in tri_token_length]
        import math
        def position(l):
            return [index for index, value in enumerate(sentences)]

        sent_position= (position(sentences))
        num_sent=len(sent_position)
        position = []
        position_rbm = []
        sent_pos1_rbm = 1
        sent_pos1 = 100
        position.append(sent_pos1)
        position_rbm.append(sent_pos1_rbm)
        for x in range(1,num_sent-1):
            s_p= ((num_sent-x)/num_sent)*100
            position.append(s_p)
            s_p_rbm = (num_sent-x)/num_sent
            position_rbm.append(s_p_rbm)
            
        sent_pos2 = 100
        sent_pos2_rbm = 1
        position.append(sent_pos2)
        position_rbm.append(sent_pos2_rbm)
        def convertToVSM(sentences):
            vocabulary = []
            for sents in sentences:
                vocabulary.extend(sents)
            vocabulary = list(set(vocabulary))
            vectors = []
            for sents in sentences:
                vector = []
                for tokenss in vocabulary:
                    vector.append(sents.count(tokenss))
                vectors.append(vector)
            return vectors
        VSM=convertToVSM(sentences)
        sentencelength=len(sentences)
        def calcMeanTF_ISF(VSM, index):
            vocab_len = len(VSM[index])
            sentences_len = len(VSM)
            count = 0
            tfisf = 0
            for i in range(vocab_len):
                tf = VSM[index][i]
                if(tf>0):
                    count += 1
                    sent_freq = 0
                    for j in range(sentences_len):
                        if(VSM[j][i]>0): sent_freq += 1
                    tfisf += (tf)*(1.0/sent_freq)
            if(count > 0):
                mean_tfisf = tfisf/count
            else:
                mean_tfisf = 0
            return tf, (1.0/sent_freq), mean_tfisf
        tfvec=[]
        isfvec=[]
        tfisfvec=[]
        tfisfvec_rbm=[]
        for i in range(sentencelength):
            x,y,z=calcMeanTF_ISF(VSM,i)
            tfvec.append(x)
            isfvec.append(y)
            tfisfvec.append(z*100)
            tfisfvec_rbm.append(z)
        maxtf_isf=max(tfisfvec_rbm)
        centroid=[]
        centroid.append(maxtf_isf)
        centroid=(max(VSM))
        from numpy import dot
        from numpy.linalg import norm
        cosine_similarity=[]
        cosine_similarity_rbm=[]
        for z in range(sentencelength):
            cos_simi = ((dot(centroid, VSM[z])/(norm(centroid)*norm(VSM[z])))*100)
            cosine_similarity.append(cos_simi)
            cos_simi_rbm = (dot(centroid, VSM[z])/(norm(centroid)*norm(VSM[z])))
            cosine_similarity_rbm.append(cos_simi_rbm)
        sent_word=[]
        for u in range(len(sentences)):
            sent_split1=[w.lower() for w in sentences[u].split(" ")]
            sent_split=[w for w in sent_split1 if w not in stop_words and w not in punctuation and not w.isdigit()]
            a=(len(sent_split))
            sent_word.append(a)
        longest_sent=max(sent_word)
        sent_length=[]
        sent_length_rbm=[]
        for x in sent_word:
            sent_length.append((x/longest_sent)*100)
            sent_length_rbm.append(x/longest_sent)

        import re
        num_word=[]
        numeric_token=[]
        numeric_token_rbm=[]
        for u in range(len(sentences)):
            sent_split4=sentences[u].split(" ")
            e=re.findall("\d+",sentences[u])
            noofwords=(len(e))
            num_word.append(noofwords)
            numeric_token.append((num_word[u]/sent_word[u])*100)
            numeric_token_rbm.append(num_word[u]/sent_word[u])
        from rake_nltk import Rake
        r = Rake() 
        keywords=[]
        for s in sentences:
            r.extract_keywords_from_text(s)
            key=list(r.get_ranked_phrases())
            keywords.append(key)
        l_keywords=[]
        for s in keywords:
            leng=len(s)
            l_keywords.append(leng)
        total_keywords=sum(l_keywords)
        thematic_number= []
        thematic_number_rbm= []
        for x in l_keywords:
            thematic_number.append((x/total_keywords)*100)
            thematic_number_rbm.append(x/total_keywords)


        from nltk.tag import pos_tag
        from collections import Counter
        pncounts = []
        pncounts_rbm = []
        for sentence in sentences:
            tagged=nltk.pos_tag(nltk.word_tokenize(str(sentence)))
            counts = Counter(tag for word,tag in tagged if tag.startswith('NNP') or tag.startswith('NNPS'))
            f=sum(counts.values())
            pncounts.append(f)
            pncounts_rbm.append(f)
        pnounscore=[(int(o) / int(p))*100 for o,p in zip(pncounts, sent_word)]
        pnounscore_rbm=[int(o) / int(p) for o,p in zip(pncounts_rbm, sent_word)]
        featureMatrix = []
        featureMatrix.append(position_rbm)
        featureMatrix.append(bi_token_length)
        featureMatrix.append(tri_token_length)
        featureMatrix.append(tfisfvec_rbm)
        featureMatrix.append(cosine_similarity_rbm)
        featureMatrix.append(thematic_number_rbm)
        featureMatrix.append(sent_length_rbm)
        featureMatrix.append(numeric_token_rbm)
        featureMatrix.append(pnounscore_rbm)



        featureMat = np.zeros((len(sentences),9))
        for i in range(9) :
            for j in range(len(sentences)):
                featureMat[j][i] = featureMatrix[i][j]
        featureMat_normed = featureMat
        np.save('output_labels_10.npy',featureMat_normed)
        import rbm

        temp = rbm.test_rbm(dataset = featureMat_normed,learning_rate=0.01, training_epochs=15, batch_size=5,n_chains=9,
                     n_hidden=9)

        enhanced_feature_sum = []

        for i in range(len(np.sum(temp,axis=1))) :
            enhanced_feature_sum.append([np.sum(temp,axis=1)[i],i])
            emptyarray[i][0][2]=np.sum(temp,axis=1)[i]

        enhanced_feature_sum.sort(key=lambda x: x[0])

        length_to_be_extracted = len(enhanced_feature_sum)/2
        extracted_sentences = []
        extracted_sentences.append([sentences[0], 0])



        for x in range(int(length_to_be_extracted)) :
            if(enhanced_feature_sum[x][1] != 0) :
                extracted_sentences.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
               
        extracted_sentences.sort(key=lambda x: x[1])
        summary1=[]
        for i in range(len(extracted_sentences)):
            final_text="\n"+extracted_sentences[i][0]
            final_text_1=extracted_sentences[i][0]
            summary1.append(final_text_1)
        emparray1 = emptyarray[0]
        emparray2 = emptyarray[1:,:,:]

        emparray2 = emparray2[np.argsort(emparray2[:,0,2])]
        emparray2 = emparray2[::-1]
        sh=emparray2.shape[0]
        sh=int(sh/2)+1
        emparray2=emparray2[:sh]
        emparray2 = emparray2[np.argsort(emparray2[:,0,1])]
        rbmarray=emparray2[:,:,:2]
        rbm_summary = []
        for i in range(rbmarray.shape[0]):
            rbm_summary.append(rbmarray[i][0][0])
        import numpy as np
        import matplotlib
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        position1 = ctrl.Antecedent(np.arange(0, 100, 10), 'position1')
        cos_similarity = ctrl.Antecedent(np.arange(0, 100, 10), 'cos_similarity')
        bitokens = ctrl.Antecedent(np.arange(0, 100, 10), 'bitokens')
        tritokens = ctrl.Antecedent(np.arange(0, 100, 10), 'tritokens')
        propernoun = ctrl.Antecedent(np.arange(0, 100, 10), 'propernoun')
        sentencelength = ctrl.Antecedent(np.arange(0, 100, 10), 'sentencelength')
        numtokens = ctrl.Antecedent(np.arange(0, 100, 10), 'numtokens')
        keywords = ctrl.Antecedent(np.arange(0, 10, 1), 'keywords')
        tf_isf = ctrl.Antecedent(np.arange(0, 100, 10), 'tf_isf')


        senten = ctrl.Consequent(np.arange(0, 100, 10), 'senten')

        position1.automf(3)
        cos_similarity.automf(3)
        bitokens.automf(3)
        tritokens.automf(3)
        propernoun.automf(3)
        sentencelength.automf(3)
        numtokens.automf(3)
        keywords.automf(3)
        tf_isf.automf(3)


        senten['bad'] = fuzz.trimf(senten.universe, [0, 0, 50])
        senten['avg'] = fuzz.trimf(senten.universe, [0, 50, 100])
        senten['good'] = fuzz.trimf(senten.universe, [50, 100, 100])

        rule1 = ctrl.Rule(position1['good'] & sentencelength['good'] & propernoun['good'] &numtokens['good'], senten['good'])
        rule2 = ctrl.Rule(position1['poor'] & sentencelength['poor'] & numtokens['poor'], senten['bad'])
        rule3 = ctrl.Rule(propernoun['poor'] & keywords['average'], senten['bad'])
        rule4 = ctrl.Rule(cos_similarity['good'], senten['good'])
        rule5 = ctrl.Rule(bitokens['good'] & tritokens['good'] & numtokens['average'] | tf_isf['average'], senten['avg'])


        sent_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5])
        Sent = ctrl.ControlSystemSimulation(sent_ctrl)
        fuzzemptyarr= np.empty((20,1,2), dtype=object)
        t2=0
        summary2=[]
        for s in range(len(senten)):
            Sent.input['position1'] = int(position[s])
            Sent.input['cos_similarity'] = int(cosine_similarity[s])
            Sent.input['bitokens'] = int(bi_tokens[s])
            Sent.input['tritokens'] = int(tri_tokens[s])
            Sent.input['tf_isf'] = int(tfisfvec[s])
            Sent.input['keywords'] = int(thematic_number[s])
            Sent.input['propernoun'] = int(pnounscore[s])
            Sent.input['sentencelength'] = int(sent_length[s])
            Sent.input['numtokens'] = int(numeric_token[s])
            Sent.compute()
            if Sent.output['senten'] > 50:
                summary2.append((sentences[s]))
                fuzzemptyarr[t2][0][0] = sentences[s]
                fuzzemptyarr[t2][0][1] = s
                t2+=1
        fuzzarray = np.empty((len(summary2),1,2),dtype=object)
        for i in range(len(summary2)):
            fuzzarray[i][0][0] = fuzzemptyarr[i][0][0]
            fuzzarray[i][0][1] = fuzzemptyarr[i][0][1]
            
        fuzzarray=fuzzarray[1:]

        summarray=np.empty((emparray2.shape), dtype=object) 
        t1=0
        for i in range(emparray2.shape[0]):
            t = emparray2[i][0][0]
            t2 =emparray2[i][0][1]
            for j in summary2:
                if t == j:
                    summarray[t1][0][0] = t
                    summarray[t1][0][1] = t2
                    t1 += 1
        commsentarray=summarray[summarray != np.array(None)]
        commonarray = np.empty((int(commsentarray.shape[0]/2),1,2),dtype=object)
        for i in range((int(commsentarray.shape[0]/2))):
            commonarray[i][0][0] = summarray[i][0][0]
            commonarray[i][0][1] = summarray[i][0][1]

        concarray=np.concatenate((rbmarray,fuzzarray), axis=0)
        concarray2 = concarray

        for i in range(commonarray.shape[0]):
            t1 = commonarray[i][0][0]
            for j in range(concarray.shape[0]):
                t2 = concarray[j][0][0]
                if t1 == t2:
                    concarray2[j][0][0] = 0
                    concarray2[j][0][1] = 0

        concarray3=concarray2[concarray2 != 0]
        uncommonarray= np.empty((int(concarray3.shape[0]/2),1,2), dtype=object)
        t1=0
        t2=0

        for i in range(concarray3.shape[0]):
            if i % 2 == 0:
                uncommonarray[t1][0][0]=concarray3[i]
                t1+=1
            else:
                uncommonarray[t2][0][1]=concarray3[i]
                t2+=1
        uncommonarray=uncommonarray[np.argsort(uncommonarray[:,0,1])]
        uncommonarray=uncommonarray[:int(uncommonarray.shape[0]/2)]
        emparray5=emparray1[0][0]
        firstsent=np.empty((1,1,2),dtype=object)
        emparray6=emparray1[0][1]
        firstsent[0][0][0]=emparray5
        firstsent[0][0][1]=emparray6
        final_summ_array=np.concatenate((firstsent,commonarray,uncommonarray))
        final_summ_array=final_summ_array[np.argsort(final_summ_array[:,0,1])]
        final_summary = []
        for i in range(int((final_summ_array.shape[0])*0.5)):
            final_summary.append(final_summ_array[i][0][0])
        print("Final Summary \n\n",final_summary)
    except:
        print("Summary \n\n",sentences)
        
from rouge import Rouge
hypothesis = ''
for s in sentences:
    hypothesis=hypothesis+' ' + s
rouge= Rouge()
print(rouge.get_scores(hypothesis,text))
