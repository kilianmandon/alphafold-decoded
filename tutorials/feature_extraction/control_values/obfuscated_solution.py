import torch #line:1
import re #line:2
from torch import nn #line:3
_OO0000OOOO0O0O0O0 =["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V",]#line:5
_OO00OO0O00000OO0O =_OO0000OOOO0O0O0O0 +["X"]#line:6
_O0O0O00000O0OO0O0 =_OO00OO0O00000OO0O +["-"]#line:7
restype_order_with_x =None #line:9
restype_order_with_x_and_gap =None #line:10
restype_order_with_x ={OO00O0OOOO0OO000O :OO00OOOOO0OOOO0OO for OO00OOOOO0OOOO0OO ,OO00O0OOOO0OO000O in enumerate (_OO00OO0O00000OO0O )}#line:17
restype_order_with_x_and_gap ={OO00O000OOOOO00O0 :OO00OO0O000OO0OOO for OO00OO0O000OO0OOO ,OO00O000OOOOO00O0 in enumerate (_O0O0O00000O0OO0O0 )}#line:18
def load_a3m_file (OO0OOO000O00OO0O0 :str ):#line:26
    ""#line:35
    with open (OO0OOO000O00OO0O0 ,'r')as OO00OO0OOO0OO00O0 :#line:45
        OOOO000OOOOO0OOO0 =OO00OO0OOO0OO00O0 .readlines ()#line:46
    O0O0O00O000OOOOOO =[O000O0O0OO0OO0000 for O000O0O0OO0OO0000 ,O0O00O0O0O0O000OO in enumerate (OOOO000OOOOO0OOO0 )if O0O00O0O0O0O000OO .startswith ('>')]#line:48
    O000O00OO0000O0O0 =[OOOO000OOOOO0OOO0 [O0000000000OO0O00 +1 ].strip ()for O0000000000OO0O00 in O0O0O00O000OOOOOO ]#line:49
    return O000O00OO0000O0O0 #line:55
def onehot_encode_aa_type (O00OOO0O0O00OOOO0 ,include_gap_token =False ):#line:59
    ""#line:71
    O000000OOO0000O00 =restype_order_with_x if not include_gap_token else restype_order_with_x_and_gap #line:72
    OO0OOOO00O00O0OO0 =None #line:73
    O00O0OO0OO0O0OOOO =torch .tensor ([O000000OOO0000O00 [O0000OO0OO0OOOO00 ]for O0000OO0OO0OOOO00 in O00OOO0O0O00OOOO0 ])#line:83
    OO0OOOO00O00O0OO0 =nn .functional .one_hot (O00O0OO0OO0O0OOOO ,num_classes =len (O000000OOO0000O00 ))#line:84
    return OO0OOOO00O00O0OO0 #line:90
def initial_data_from_seqs (O0000O0O000OOO0O0 ):#line:94
    ""#line:116
    OOO00OOO00000OOOO =None #line:118
    O0O0O000O00O0OOOO =None #line:119
    OO0O0000O000000O0 =None #line:120
    O0O0O000O00O0OOOO =[]#line:145
    OOO00OOO00000OOOO =[]#line:146
    for O000OOOOOOOOOOOOO in O0000O0O000OOO0O0 :#line:147
        O000OOOOO0OO0000O =[]#line:148
        OOOO00O0O00OO0O00 =0 #line:149
        for O0OOOO00O0OO0OO0O in O000OOOOOOOOOOOOO :#line:150
            if O0OOOO00O0OO0OO0O .islower ():#line:151
                OOOO00O0O00OO0O00 +=1 #line:152
            else :#line:153
                O000OOOOO0OO0000O .append (OOOO00O0O00OO0O00 )#line:154
                OOOO00O0O00OO0O00 =0 #line:155
        O0OOO0O0OOOO0O000 =re .sub ('[a-z]','',O000OOOOOOOOOOOOO )#line:156
        if O0OOO0O0OOOO0O000 in OOO00OOO00000OOOO :#line:158
            continue #line:159
        OOO00OOO00000OOOO .append (O0OOO0O0OOOO0O000 )#line:161
        O0O0O000O00O0OOOO .append (O000OOOOO0OO0000O )#line:162
    OOO00OOO00000OOOO =torch .stack ([onehot_encode_aa_type (O00OO000O0OO0O000 ,include_gap_token =True )for O00OO000O0OO0O000 in OOO00OOO00000OOOO ],dim =0 )#line:164
    OOO00OOO00000OOOO =OOO00OOO00000OOOO .float ()#line:165
    O0O0O000O00O0OOOO =torch .tensor (O0O0O000O00O0OOOO ).float ()#line:166
    OO0O0000O000000O0 =OOO00OOO00000OOOO .float ().mean (dim =0 )#line:167
    return {'msa_aatype':OOO00OOO00000OOOO ,'msa_deletion_count':O0O0O000O00O0OOOO ,'aa_distribution':OO0O0000O000000O0 }#line:173
def select_cluster_centers (O0OO00O0OO0O0O0OO ,max_msa_clusters =512 ,seed =None ):#line:175
    ""#line:192
    OO0OOO00O00OO0OO0 ,O0O000O00O0OO0000 =O0OO00O0OO0O0O0OO ['msa_aatype'].shape [:2 ]#line:194
    OOOO0O0O0OOO0O0OO =['msa_aatype','msa_deletion_count']#line:195
    max_msa_clusters =min (max_msa_clusters ,OO0OOO00O00OO0OO0 )#line:196
    OOO00000000O000OO =None #line:198
    if seed is not None :#line:199
        OOO00000000O000OO =torch .Generator (O0OO00O0OO0O0O0OO ['msa_aatype'].device )#line:200
        OOO00000000O000OO .manual_seed (seed )#line:201
    O00O0OOO00000O000 =torch .randperm (OO0OOO00O00OO0OO0 -1 ,generator =OOO00000000O000OO )+1 #line:217
    O00O0OOO00000O000 =torch .cat ((torch .tensor ([0 ]),O00O0OOO00000O000 ),dim =0 )#line:218
    for O0000O00O0000OOOO in OOOO0O0O0OOO0O0OO :#line:220
        O00O00O00OOO0O0OO =f'extra_{O0000O00O0000OOOO}'#line:221
        OOO0OO0O0O000000O =O0OO00O0OO0O0O0OO [O0000O00O0000OOOO ]#line:222
        O0OO00O0OO0O0O0OO [O00O00O00OOO0O0OO ]=OOO0OO0O0O000000O [O00O0OOO00000O000 [max_msa_clusters :]]#line:223
        O0OO00O0OO0O0O0OO [O0000O00O0000OOOO ]=OOO0OO0O0O000000O [O00O0OOO00000O000 [:max_msa_clusters ]]#line:224
    return O0OO00O0OO0O0O0OO #line:230
def mask_cluster_centers (O0OOOO0OOOOO00OO0 ,mask_probability =0.15 ,seed =None ):#line:232
    ""#line:253
    O0O0O000000OO0OO0 ,O00000O000O000OO0 =O0OOOO0OOOOO00OO0 ['msa_aatype'].shape [:2 ]#line:255
    O0OOO0OO0O000OOO0 =23 #line:256
    O0O0OO0OO0OO0O00O ={'uniform_replacement':0.1 ,'replacement_from_distribution':0.1 ,'no_replacement':0.1 ,'masked_out':0.7 ,}#line:262
    O0O000O0O0O000O00 =None #line:263
    if seed is not None :#line:264
        O0O000O0O0O000O00 =torch .Generator (O0OOOO0OOOOO00OO0 ['msa_aatype'].device )#line:265
        O0O000O0O0O000O00 .manual_seed (seed )#line:266
        torch .manual_seed (seed )#line:267
    O0O0OO00OO00000OO =torch .tensor ([1 /20 ]*20 +[0 ,0 ])*O0O0OO0OO0OO0O00O ['uniform_replacement']#line:296
    OOO0000O0OO00O00O =O0OOOO0OOOOO00OO0 ['aa_distribution']*O0O0OO0OO0OO0O00O ['replacement_from_distribution']#line:298
    OO0OOOOO0OO0OO0OO =O0OOOO0OOOOO00OO0 ['msa_aatype']*O0O0OO0OO0OO0O00O ['no_replacement']#line:300
    OO0O0OOOO0OOO0O0O =torch .ones ((O0O0O000000OO0OO0 ,O00000O000O000OO0 ,1 ))*O0O0OO0OO0OO0O00O ['masked_out']#line:302
    O0O0OO00OO00000OO =O0O0OO00OO00000OO [None ,None ,...].broadcast_to (OO0OOOOO0OO0OO0OO .shape )#line:304
    OOO0000O0OO00O00O =OOO0000O0OO00O00O [None ,...].broadcast_to (OO0OOOOO0OO0OO0OO .shape )#line:305
    OO00000O000O0OO0O =O0O0OO00OO00000OO +OOO0000O0OO00O00O +OO0OOOOO0OO0OO0OO #line:307
    OOOOOOOOO0O0000OO =torch .cat ((OO00000O000O0OO0O ,OO0O0OOOO0OOO0O0O ),dim =-1 )#line:308
    OOOOOOOOO0O0000OO =OOOOOOOOO0O0000OO .reshape (-1 ,O0OOO0OO0O000OOO0 )#line:309
    O0OO0O000O0O0O0O0 =torch .distributions .Categorical (OOOOOOOOO0O0000OO ).sample ()#line:311
    O0OO0O000O0O0O0O0 =nn .functional .one_hot (O0OO0O000O0O0O0O0 ,num_classes =O0OOO0OO0O000OOO0 )#line:312
    O0OO0O000O0O0O0O0 =O0OO0O000O0O0O0O0 .reshape (O0O0O000000OO0OO0 ,O00000O000O000OO0 ,O0OOO0OO0O000OOO0 )#line:313
    O0OO0O000O0O0O0O0 =O0OO0O000O0O0O0O0 .float ()#line:314
    OOO0O0OO00OOOO0OO =torch .rand ((O0O0O000000OO0OO0 ,O00000O000O000OO0 ),generator =O0O000O0O0O000O00 )<mask_probability #line:316
    O0OOOO0OOOOO00OO0 ['true_msa_aatype']=O0OOOO0OOOOO00OO0 ['msa_aatype'].clone ()#line:318
    OOO0O0O00O00O0OOO =torch .zeros ((O0O0O000000OO0OO0 ,O00000O000O000OO0 ,1 ))#line:319
    O0OOOO0OOOOO00OO0 ['msa_aatype']=torch .cat ((O0OOOO0OOOOO00OO0 ['msa_aatype'],OOO0O0O00O00O0OOO ),dim =-1 )#line:320
    O0OOOO0OOOOO00OO0 ['msa_aatype'][OOO0O0OO00OOOO0OO ]=O0OO0O000O0O0O0O0 [OOO0O0OO00OOOO0OO ]#line:321
    return O0OOOO0OOOOO00OO0 #line:327
def cluster_assignment (OO00000O0O0000O0O ):#line:329
    ""#line:344
    OO0000OO0O00O0OO0 ,OOO000OO00O0OOOO0 =OO00000O0O0000O0O ['msa_aatype'].shape [:2 ]#line:346
    OOOO00O0O0000OOO0 =OO00000O0O0000O0O ['extra_msa_aatype'].shape [0 ]#line:347
    OO0OOOO0O000OOOOO =OO00000O0O0000O0O ['msa_aatype'][...,:21 ]#line:366
    OOOO0OO0O0O0O0O0O =OO00000O0O0000O0O ['extra_msa_aatype'][...,:21 ]#line:367
    OOO0OOO000OOO0OOO =torch .einsum ('cra,era->ce',OO0OOOO0O000OOOOO ,OOOO0OO0O0O0O0O0O )#line:368
    O0O0O00OO00000000 =torch .argmax (OOO0OOO000OOO0OOO ,dim =0 )#line:369
    OO00000O0O0000O0O ['cluster_assignment']=O0O0O00OO00000000 #line:370
    O0OO00O00O0OOO000 =torch .bincount (O0O0O00OO00000000 ,minlength =OO0000OO0O00O0OO0 )#line:372
    OO00000O0O0000O0O ['cluster_assignment_counts']=O0OO00O00O0OOO000 #line:373
    return OO00000O0O0000O0O #line:379
def cluster_average (O0O000000O0O00OO0 ,OO0OO0O0OOO00OOOO ,OO0OO0000O0O000O0 ,OOOO00000OOO0OO00 ):#line:381
    ""#line:401
    O0O0OOOOOOOO00O0O ,OO00OOO0OO00O0OO0 =O0O000000O0O00OO0 .shape [:2 ]#line:402
    O0O0000O00OO0OOOO =OO0OO0O0OOO00OOOO .shape [0 ]#line:403
    O00O0OOOO0O0OO0O0 =(O0O0000O00OO0OOOO ,)+(1 ,)*(OO0OO0O0OOO00OOOO .dim ()-1 )#line:417
    O000O0OOOOOO0OOOO =(O0O0OOOOOOOO00O0O ,)+(1 ,)*(O0O000000O0O00OO0 .dim ()-1 )#line:418
    OO0OO0000O0O000O0 =OO0OO0000O0O000O0 .view (O00O0OOOO0O0OO0O0 ).broadcast_to (OO0OO0O0OOO00OOOO .shape )#line:420
    O0OOO0000O0O0O00O =torch .scatter_add (O0O000000O0O00OO0 ,dim =0 ,index =OO0OO0000O0O000O0 ,src =OO0OO0O0OOO00OOOO )#line:421
    OOOO00000OOO0OO00 =OOOO00000OOO0OO00 .view (O000O0OOOOOO0OOOO ).broadcast_to (O0O000000O0O00OO0 .shape )#line:422
    OO00O00O0OOO0O0O0 =O0OOO0000O0O0O00O /(OOOO00000OOO0OO00 +1 )#line:423
    return OO00O00O0OOO0O0O0 #line:429
def summarize_clusters (OOOOO00000000O0O0 ):#line:433
    ""#line:446
    O00OOO0O00OO000O0 ,O0OO0O000O000OOOO =OOOOO00000000O0O0 ['msa_aatype'].shape [:2 ]#line:448
    O0OO00OOO00OO0000 =OOOOO00000000O0O0 ['extra_msa_aatype'].shape [0 ]#line:449
    OO0O000000OOO0O0O =cluster_average (OOOOO00000000O0O0 ['msa_deletion_count'],OOOOO00000000O0O0 ['extra_msa_deletion_count'],OOOOO00000000O0O0 ['cluster_assignment'],OOOOO00000000O0O0 ['cluster_assignment_counts'])#line:467
    OO0O000000OOO0O0O =2 /torch .pi *torch .arctan (OO0O000000OOO0O0O /3 )#line:469
    OOOOOOO0OO0OOOOO0 =OOOOO00000000O0O0 ['extra_msa_aatype']#line:470
    O0O0OO000OOO0OO0O =torch .zeros (OOOOOOO0OO0OOOOO0 .shape [:-1 ]+(1 ,),dtype =OOOOOOO0OO0OOOOO0 .dtype ,device =OOOOOOO0OO0OOOOO0 .device )#line:471
    O00O0OO0OO0O0OO00 =torch .cat ((OOOOOOO0OO0OOOOO0 ,O0O0OO000OOO0OO0O ),dim =-1 )#line:472
    OOO0O0O00OOO0000O =cluster_average (OOOOO00000000O0O0 ['msa_aatype'],O00O0OO0OO0O0OO00 ,OOOOO00000000O0O0 ['cluster_assignment'],OOOOO00000000O0O0 ['cluster_assignment_counts'])#line:479
    OOOOO00000000O0O0 ['cluster_deletion_mean']=OO0O000000OOO0O0O #line:481
    OOOOO00000000O0O0 ['cluster_profile']=OOO0O0O00OOO0000O #line:482
    return OOOOO00000000O0O0 #line:488
def crop_extra_msa (O0OO0O00O00000O00 ,max_extra_msa_count =5120 ,seed =None ):#line:490
    ""#line:504
    O0O0O0OOO0O000OO0 =O0OO0O00O00000O00 ['extra_msa_aatype'].shape [0 ]#line:506
    OO0000OOOO0000000 =None #line:507
    if seed is not None :#line:508
        OO0000OOOO0000000 =torch .Generator (O0OO0O00O00000O00 ['extra_msa_aatype'].device )#line:509
        OO0000OOOO0000000 .manual_seed (seed )#line:510
    max_extra_msa_count =min (max_extra_msa_count ,O0O0O0OOO0O000OO0 )#line:512
    OO0OO0OO00O000O0O =torch .randperm (O0O0O0OOO0O000OO0 ,generator =OO0000OOOO0000000 )[:max_extra_msa_count ]#line:525
    for OO0O0OOO00OO0O0OO in O0OO0O00O00000O00 .keys ():#line:526
        if OO0O0OOO00OO0O0OO .startswith ('extra_'):#line:527
            O0OO0O00O00000O00 [OO0O0OOO00OO0O0OO ]=O0OO0O00O00000O00 [OO0O0OOO00OO0O0OO ][OO0OO0OO00O000O0O ]#line:528
    return O0OO0O00O00000O00 #line:534
def calculate_msa_feat (O0000O000O0O000O0 ):#line:536
    ""#line:546
    OO0O0O00OOO0OOOOO ,O0OOO000O00OOOOOO =O0000O000O0O000O0 ['msa_aatype'].shape [:2 ]#line:548
    O00O0O0OOOOOO0O0O =None #line:549
    OOOO0OO000OO0000O =O0000O000O0O000O0 ['msa_aatype']#line:572
    OO0OO00O00000OO00 =(O0000O000O0O000O0 ['msa_deletion_count']>0 ).float ().unsqueeze (-1 )#line:574
    O000O0OOO0000O0O0 =2 /torch .pi *torch .arctan (O0000O000O0O000O0 ['msa_deletion_count']/3 )#line:576
    O000O0OOO0000O0O0 =O000O0OOO0000O0O0 .unsqueeze (-1 )#line:577
    OO00O00OOOOOOO0OO =O0000O000O0O000O0 ['cluster_deletion_mean'].unsqueeze (-1 )#line:579
    O0OO0O0000OOOO0O0 =O0000O000O0O000O0 ['cluster_profile']#line:580
    O00O0O0OOOOOO0O0O =torch .cat ((OOOO0OO000OO0000O ,OO0OO00O00000OO00 ,O000O0OOO0000O0O0 ,O0OO0O0000OOOO0O0 ,OO00O00OOOOOOO0OO ),dim =-1 )#line:582
    return O00O0O0OOOOOO0O0O #line:588
def calculate_extra_msa_feat (OO0000O0000OO0O00 ):#line:590
    ""#line:601
    OOO00O00OOO0OOO00 ,O0O00OOOO000O00O0 =OO0000O0000OO0O00 ['extra_msa_aatype'].shape [:2 ]#line:603
    OOOOOO0O0OOOO0O0O =None #line:604
    OO00OOO0000O000OO =torch .zeros ((OOO00O00OOO0OOO00 ,O0O00OOOO000O00O0 ,1 ))#line:625
    OO0OO00OOO0000OO0 =torch .cat ((OO0000O0000OO0O00 ['extra_msa_aatype'],OO00OOO0000O000OO ),dim =-1 )#line:626
    OOOOOO0O0O0OO0O00 =(OO0000O0000OO0O00 ['extra_msa_deletion_count']>0 ).float ().unsqueeze (-1 )#line:627
    O000O0OOOO0O0000O =2 /torch .pi *torch .arctan (OO0000O0000OO0O00 ['extra_msa_deletion_count']/3 )#line:628
    O000O0OOOO0O0000O =O000O0OOOO0O0000O .unsqueeze (-1 )#line:629
    OOOOOO0O0OOOO0O0O =torch .cat ((OO0OO00OOO0000OO0 ,OOOOOO0O0O0OO0O00 ,O000O0OOOO0O0000O ),dim =-1 )#line:631
    return OOOOOO0O0OOOO0O0O #line:637
def create_features_from_a3m (O0OOO0O0O00O0OOO0 ,seed =None ):#line:641
    ""#line:658
    O0OO00O000O0O00O0 =None #line:660
    OO0O0OO0O0O0OO0OO =None #line:661
    OO0O0000OOOO0O000 =None #line:662
    OO000OOOOO0O0OO0O =None #line:663
    OOO0OOOOOO00OO0OO =None #line:664
    OOOOOOOO0OOO0OOOO =None #line:665
    OOO000OO00O000000 =None #line:666
    if seed is not None :#line:667
        OOO0OOOOOO00OO0OO =seed #line:668
        OOOOOOOO0OOO0OOOO =seed +1 #line:669
        OOO000OO00O000000 =seed +2 #line:670
    O0O00OOO0OOOO0O00 =load_a3m_file (O0OOO0O0O00O0OOO0 )#line:693
    OOO00O0OOOO00OO0O =initial_data_from_seqs (O0O00OOO0OOOO0O00 )#line:694
    OO00OO00O0000O0OO =[lambda O0OO0OO000OOOOO00 :select_cluster_centers (O0OO0OO000OOOOO00 ,seed =OOO0OOOOOO00OO0OO ),lambda OOO0OOO0OOOOOOOO0 :mask_cluster_centers (OOO0OOO0OOOOOOOO0 ,seed =OOOOOOOO0OOO0OOOO ),cluster_assignment ,summarize_clusters ,lambda O0O000000O000O00O :crop_extra_msa (O0O000000O000O00O ,seed =OOO000OO00O000000 )]#line:702
    for O00O0OOOO00O0O00O in OO00OO00O0000O0OO :#line:704
        OOO00O0OOOO00OO0O =O00O0OOOO00O0O00O (OOO00O0OOOO00OO0O )#line:705
    O0OO00O000O0O00O0 =calculate_msa_feat (OOO00O0OOOO00OO0O )#line:707
    OO0O0OO0O0O0OO0OO =calculate_extra_msa_feat (OOO00O0OOOO00OO0O )#line:708
    OO0O0000OOOO0O000 =onehot_encode_aa_type (O0O00OOO0OOOO0O00 [0 ],include_gap_token =False ).float()#line:712
    OO000OOOOO0O0OO0O =torch .arange (len (O0O00OOO0OOOO0O00 [0 ]))#line:713
    return {'msa_feat':O0OO00O000O0O00O0 ,'extra_msa_feat':OO0O0OO0O0O0OO0OO ,'target_feat':OO0O0000OOOO0O000 ,'residue_index':OO000OOOOO0O0OO0O }#line:724
def create_control_values (O0O00OOO00O000O00 ):#line:726
    O0O00000OOOO0OOOO =f'{O0O00OOO00O000O00}/alignment_tautomerase.a3m'#line:727
    OOOOO00O0O0O0O0O0 =f'{O0O00OOO00O000O00}/control_values'#line:728
    O0O0OOO0O000O0O0O =load_a3m_file (O0O00000OOOO0OOOO )#line:730
    OO0O00O0000O00O0O =initial_data_from_seqs (O0O0OOO0O000O0O0O )#line:732
    torch .save (OO0O00O0000O00O0O ,f'{OOOOO00O0O0O0O0O0}/initial_data.pt')#line:733
    OO0OO000OO0O0O00O =select_cluster_centers (OO0O00O0000O00O0O ,seed =0 )#line:734
    torch .save (OO0OO000OO0O0O00O ,f'{OOOOO00O0O0O0O0O0}/clusters_selected.pt')#line:735
    O0O000OO0OO0OOOO0 =mask_cluster_centers (OO0OO000OO0O0O00O ,seed =1 )#line:736
    torch .save (O0O000OO0OO0OOOO0 ,f'{OOOOO00O0O0O0O0O0}/clusters_masked.pt')#line:737
    O0OO00O0OO0OOOOO0 =cluster_assignment (O0O000OO0OO0OOOO0 )#line:738
    torch .save (O0OO00O0OO0OOOOO0 ,f'{OOOOO00O0O0O0O0O0}/clusters_assigned.pt')#line:739
    OOOO0OOO00OOO000O =summarize_clusters (O0OO00O0OO0OOOOO0 )#line:740
    torch .save (OOOO0OOO00OOO000O ,f'{OOOOO00O0O0O0O0O0}/clusters_summarized.pt')#line:741
    O00000O00O0OOO0O0 =crop_extra_msa (OOOO0OOO00OOO000O ,seed =2 )#line:742
    torch .save (O00000O00O0OOO0O0 ,f'{OOOOO00O0O0O0O0O0}/extra_msa_cropped.pt')#line:743
    OO0OOO0OOOO00OOOO =calculate_msa_feat (O00000O00O0OOO0O0 )#line:745
    OO000000O00O00000 =calculate_extra_msa_feat (O00000O00O0OOO0O0 )#line:746
    torch .save (OO0OOO0OOOO00OOOO ,f'{OOOOO00O0O0O0O0O0}/msa_feat.pt')#line:747
    torch .save (OO000000O00O00000 ,f'{OOOOO00O0O0O0O0O0}/extra_msa_feat.pt')#line:748
    O0000OO000O0O0OOO =create_features_from_a3m (O0O00000OOOO0OOOO ,seed =0 )#line:751
    torch .save (O0000OO000O0O0OOO ,f'{OOOOO00O0O0O0O0O0}/full_batch.pt')#line:752