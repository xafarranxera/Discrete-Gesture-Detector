import time
import pandas as pd
import math
from sklearn.svm import LinearSVC
from sklearn.metrics import *
import numpy as np
import pickle
import os

# -------------------- PARÀMETRES ----------------------

df_header_model =   ["MiddleTip_x", "MiddleTip_y", "MiddleTip_z",
                     "Palm_x", "Palm_y", "Palm_z",
                     "RingMiddleJoint_x", "RingMiddleJoint_y", "RingMiddleJoint_z",
                     "ThumbTip_x", "ThumbTip_y", "ThumbTip_z",
                     "PinkyDistalJoint_x", "PinkyDistalJoint_y", "PinkyDistalJoint_z",
                     "ThumbProximalJoint_x", "ThumbProximalJoint_y", "ThumbProximalJoint_z",
                     "IndexKnuckle_x", "IndexKnuckle_y", "IndexKnuckle_z",
                     "IndexMiddleJoint_x", "IndexMiddleJoint_y", "IndexMiddleJoint_z",
                     "PinkyTip_x", "PinkyTip_y", "PinkyTip_z",
                     "RingMetacarpal_x", "RingMetacarpal_y", "RingMetacarpal_z",
                     "MiddleDistalJoint_x", "MiddleDistalJoint_y", "MiddleDistalJoint_z",
                     "RingTip_x", "RingTip_y", "RingTip_z",
                     "IndexMetacarpal_x", "IndexMetacarpal_y", "IndexMetacarpal_z",
                     "RingDistalJoint_x", "RingDistalJoint_y", "RingDistalJoint_z",
                     "MiddleMiddleJoint_x", "MiddleMiddleJoint_y", "MiddleMiddleJoint_z",
                     "ThumbMetacarpalJoint_x", "ThumbMetacarpalJoint_y", "ThumbMetacarpalJoint_z",
                     "IndexTip_x", "IndexTip_y", "IndexTip_z",
                     "PinkyKnuckle_x", "PinkyKnuckle_y", "PinkyKnuckle_z",
                     "MiddleMetacarpal_x", "MiddleMetacarpal_y", "MiddleMetacarpal_z",
                     "PinkyMetacarpal_x", "PinkyMetacarpal_y", "PinkyMetacarpal_z",
                     "ThumbDistalJoint_x", "ThumbDistalJoint_y", "ThumbDistalJoint_z",
                     "IndexDistalJoint_x", "IndexDistalJoint_y", "IndexDistalJoint_z",
                     "RingKnuckle_x", "RingKnuckle_y", "RingKnuckle_z", 
                     "PinkyMiddleJoint_x", "PinkyMiddleJoint_y", "PinkyMiddleJoint_z"]
    
origin_joint = "Wrist"
dest_joint = "MiddleKnuckle"

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, 'discrete_gesture_model.sav')
#model_path = "/home/vortiz/MicroSD-128/ITI/2022/LYNX/discrete_gesture_model.sav"

text = "3201 2222222 21354234 \
        ThumbProximalJoint 1.0123 2.0123 3.0123 \
        PinkyMiddleJoint 2.0123 3.0123 4.0123 \
        RingMetacarpal 3.0123 4.0123 5.0123 \
        MiddleMetacarpal 4.0123 5.0123 6.0123 \
        ThumbMetacarpalJoint 5.0123 6.0123 7.0123 \
        IndexMiddleJoint 6.0123 7.0123 8.0123 \
        PinkyMetacarpal 7.0123 8.0123 9.0123 \
        IndexTip 8.0123 9.0123 10.0123 \
        RingTip 9.0123 10.0123 11.0123 \
        MiddleMiddleJoint 10.0123 11.0123 12.0123 \
        MiddleTip 11.0123 12.0123 13.0123 \
        RingKnuckle 12.0123 13.0123 14.0123 \
        MiddleDistalJoint 13.0123 14.0123 15.0123 \
        ThumbTip 14.0123 15.0123 16.0123 \
        PinkyDistalJoint 15.0123 16.0123 17.0123 \
        IndexDistalJoint 16.0123 17.0123 18.0123 \
        PinkyKnuckle 17.0123 18.0123 19.0123 \
        ThumbDistalJoint 18.0123 19.0123 20.0123 \
        RingMiddleJoint 19.0123 20.0123 21.0123 \
        Palm 20.0123 21.0123 22.0123 \
        RingDistalJoint 21.0123 22.0123 23.0123 \
        IndexKnuckle 22.0123 23.0123 24.0123 \
        PinkyTip 23.0123 24.0123 25.0123 \
        IndexMetacarpal 24.0123 25.0123 26.0123 \
        MiddleKnuckle 30.0123 50.0123 40.0123 \
        Wrist 55.0123 44.0123 66.0123"

trial_cols = ["MiddleTip_x", "MiddleTip_y", "MiddleTip_z", "Palm_x", "Palm_y", "Palm_z", "RingMiddleJoint_x", "RingMiddleJoint_y", "RingMiddleJoint_z", "ThumbTip_x", "ThumbTip_y", "ThumbTip_z", "PinkyDistalJoint_x", "PinkyDistalJoint_y", "PinkyDistalJoint_z", "ThumbProximalJoint_x", "ThumbProximalJoint_y", "ThumbProximalJoint_z", "IndexKnuckle_x", "IndexKnuckle_y", "IndexKnuckle_z", "IndexMiddleJoint_x", "IndexMiddleJoint_y", "IndexMiddleJoint_z", "MiddleKnuckle_x", "MiddleKnuckle_y", "MiddleKnuckle_z", "PinkyTip_x", "PinkyTip_y", "PinkyTip_z", "RingMetacarpal_x", "RingMetacarpal_y", "RingMetacarpal_z", "MiddleDistalJoint_x", "MiddleDistalJoint_y", "MiddleDistalJoint_z", "RingTip_x", "RingTip_y", "RingTip_z", "IndexMetacarpal_x", "IndexMetacarpal_y", "IndexMetacarpal_z", "RingDistalJoint_x", "RingDistalJoint_y", "RingDistalJoint_z", "MiddleMiddleJoint_x", "MiddleMiddleJoint_y", "MiddleMiddleJoint_z", "ThumbMetacarpalJoint_x", "ThumbMetacarpalJoint_y", "ThumbMetacarpalJoint_z", "IndexTip_x", "IndexTip_y", "IndexTip_z", "PinkyKnuckle_x", "PinkyKnuckle_y", "PinkyKnuckle_z", "MiddleMetacarpal_x", "MiddleMetacarpal_y", "MiddleMetacarpal_z", "PinkyMetacarpal_x", "PinkyMetacarpal_y", "PinkyMetacarpal_z", "ThumbDistalJoint_x", "ThumbDistalJoint_y", "ThumbDistalJoint_z", "IndexDistalJoint_x", "IndexDistalJoint_y", "IndexDistalJoint_z", "RingKnuckle_x", "RingKnuckle_y", "RingKnuckle_z", "PinkyMiddleJoint_x", "PinkyMiddleJoint_y", "PinkyMiddleJoint_z", "Wrist_x", "Wrist_y", "Wrist_z"]

# ---------------------- FUNCIONS ---------------------------

def rotate(elem, azim_0, elev_0):
    """Apply elevation and azimuth rotations on each joint"""

    # Calculate elevation and azimuth for each frame
    [X, Z, Y] = elem    # MIND the swap
    [X_new, Y_new, Z_new] = [X, Y, Z]
    
    # Prevent processing the origin (zeros)
    if not (X==0 and Y==0 and Z==0):
        XY = math.sqrt(X**2 + Y**2)
        elev = np.arctan(Z / XY)
        azim = np.arctan2(X, Y)
        new_elev = elev - elev_0
        new_azim = azim - azim_0
        dist = math.sqrt(X**2 + Y**2 + Z**2)
        Z_new = dist * math.sin(new_elev)
        XY_new = dist * math.cos(new_elev)
        X_new = XY_new * math.cos(new_azim)
        Y_new = XY_new * math.sin(new_azim)
    
    return [X_new, Z_new, Y_new]    # MIND the swap
    
def normalise_elev_and_azim(df, origin_joint, dest_joint):
    """Calculate elevation and azimuth correction"""    
    
    for i, row in df.iterrows():    # No caldria, atés que cada df té només una fila (mesura)

        # Calculate elevation and azimuth for each frame's reference
        diff_X = row[dest_joint+"_x"] - row[origin_joint+"_x"]
        diff_Y = row[dest_joint+"_z"] - row[origin_joint+"_z"]  # MIND the swap
        diff_Z = row[dest_joint+"_y"] - row[origin_joint+"_y"]  # MIND the swap

        diff_XY = math.sqrt(diff_X**2 + diff_Y**2)
        elev_0 = np.arctan(diff_Z / diff_XY)
        azim_0 = np.arctan2(diff_X, diff_Y)

        for j, elem in enumerate(row):
            if j % 3 == 0:
                df.iloc[i,j], df.iloc[i,j+1], df.iloc[i,j+2]= rotate([elem, df.iloc[i,j+1], df.iloc[i,j+2]], azim_0, elev_0)

        # Create elevation and azimuth columns
        df["elev"], df["azim"] = elev_0, azim_0

    return df

def normalise_translation(df, origin_joint):
    """Calculate and apply translation to origin_joint"""    

    df_aux = df
    columns = list(df.columns)
    for col in columns:
        coord = col.split("_")[1]   # x, y or z
        df_aux[col] = df[col] - df[origin_joint+"_"+coord]

    return df_aux

def get_joint_data_from_string(text):
    """Parse data from UDP plain string""" 

    frame_dict, coord_frame_dict = {}, {}
    text_list = text.split()
    nframe, timestamp, ticks = text_list[0:3]
    del text_list[0:3]

    for i, elem in enumerate(text_list):
        if i % 4 == 0:
            frame_dict[elem] = [float(text_list[j]) for j in range(i+1,i+4)]

    for elem in frame_dict:
        coord_frame_dict[elem+"_x"], coord_frame_dict[elem+"_y"], coord_frame_dict[elem+"_z"] = frame_dict[elem]

    return nframe, timestamp, ticks, coord_frame_dict
    
def get_gesture(text, model):
    """Classify the gesture""" 

    # Comencem a comptar el temps
    start_time = time.time()

    # Extrau metadades
    nframe, _, _, coord_frame_dict = get_joint_data_from_string(text) # nframe, timestamp, ticks, coord_frame_dict
    print("-> The", nframe, "data has been received")  
    
    # Elabora dataframe a partir del diccionari (1.3 ms)
    df = pd.DataFrame([coord_frame_dict])

    # Aplica translació fins a l'articulació origin_joint (10 ms)
    df = normalise_translation(df, origin_joint)

    # Aplica rotació per a normalitzar en elevació i azímut (3 ms)
    df = normalise_elev_and_azim(df, origin_joint, dest_joint)

    # Lleva les columnes que el model no empra (0.9 ms)
    delete_cols = [col for col in list(df.columns) if origin_joint in col or dest_joint in col] + ["elev", "azim"]
    elevation = df["elev"][0]*180/math.pi
    df = df.drop(columns=delete_cols)

    # Reordena les columnes segons es va entrenar el model (0.3 ms)
    df = df[df_header_model]

    # Passa pel model inferència del df complet (1.4 ms)
    detected_gesture = model.predict(df)

    # Separa entre Empty i Stop per elevació
    if detected_gesture == "1_Empty" and elevation > 60:
        detected_gesture == "2_Palm_up"

    # Mesurem el temps que ha calgut
    elapsed_time = time.time() - start_time

    return nframe, detected_gesture, elapsed_time

if (__name__ == '__main__'):

    # ------------------- RUN ---------------------

    # Carrega el model (0.1 ms)
    model = pickle.load(open(model_path, "rb"))

    # Ordre d'inferència: preprocessa les dades i classifica el gest de nframe (15-20 ms)
    # nframe, detected_gesture = get_gesture(text, model)

    # print("Detected gesture:", detected_gesture)


    # ----------------- TRIALS -------------------

    empty_values = [0.1436558650007167, -0.0404603448339937, -0.0290891551333165, 0.0473728961765844, -0.0056065327122036, 0.0040153871244936, 0.1084911937951536, -0.0295025445524268, 0.0058370864414099, 0.0926702224219552, -0.0177578615777059, -0.0640030352444505, 0.100504427468892, -0.0488573321286419, 0.0232078202548578, 0.0518637799960734, -0.0055429754692762, -0.0434265355146584, 0.0771089953362422, 0.0086316396681813, -0.0188801915266783, 0.1146759934478472, 0.0027467676195709, -0.0325454801420009, 0.0768605026284632, 0.0, 0.0, 0.1050510946395624, -0.0579681886541337, 0.0145798292911289, 0.0181827638179771, -0.008624677625845, 0.0064745793369076, 0.1351590914817085, -0.0297285679220279, -0.021717428467581, 0.1211473086560101, -0.0580052069098164, -0.0159120150645603, 0.0192867503383536, -0.0009180735455999, -0.008536641046154, 0.1173654135748225, -0.0451058235294344, -0.0057997042461126, 0.118788360037977, -0.0150585062326427, -0.0114321623737325, 0.0181669721281921, 0.0010263225035579, -0.0183740901429752, 0.1478307108170193, -0.0076176469893757, -0.0446551700023643, 0.0662866734627584, -0.0272394237003555, 0.0252435182515066, 0.0186395699243008, -0.0039595709776258, -0.0019252675554949, 0.0163511428329989, -0.0147142903138952, 0.0119914994222272, 0.0786107089730207, -0.0165542246804423, -0.0518846042108683, 0.1351311075153464, -0.0031523015081974, -0.0404514992802158, 0.0734594165626635, -0.013769928040488, 0.0128699400803909, 0.0902065966656726, -0.0357037896469376, 0.0278324049067612, 0.0, 0.0, 0.0]
    palmup_values = [0.1436558650007167, -0.0404603448339937, -0.0290891551333165, 0.0473728961765844, -0.0056065327122036, 0.0040153871244936, 0.1084911937951536, -0.0295025445524268, 0.0058370864414099, 0.0926702224219552, -0.0177578615777059, -0.0640030352444505, 0.100504427468892, -0.0488573321286419, 0.0232078202548578, 0.0518637799960734, -0.0055429754692762, -0.0434265355146584, 0.0771089953362422, 0.0086316396681813, -0.0188801915266783, 0.1146759934478472, 0.0027467676195709, -0.0325454801420009, 0.0768605026284632, 0.0, 0.0, 0.1050510946395624, -0.0579681886541337, 0.0145798292911289, 0.0181827638179771, -0.008624677625845, 0.0064745793369076, 0.1351590914817085, -0.0297285679220279, -0.021717428467581, 0.1211473086560101, -0.0580052069098164, -0.0159120150645603, 0.0192867503383536, -0.0009180735455999, -0.008536641046154, 0.1173654135748225, -0.0451058235294344, -0.0057997042461126, 0.118788360037977, -0.0150585062326427, -0.0114321623737325, 0.0181669721281921, 0.0010263225035579, -0.0183740901429752, 0.1478307108170193, -0.0076176469893757, -0.0446551700023643, 0.0662866734627584, -0.0272394237003555, 0.0252435182515066, 0.0186395699243008, -0.0039595709776258, -0.0019252675554949, 0.0163511428329989, -0.0147142903138952, 0.0119914994222272, 0.0786107089730207, -0.0165542246804423, -0.0518846042108683, 0.1351311075153464, -0.0031523015081974, -0.0404514992802158, 0.0734594165626635, -0.013769928040488, 0.0128699400803909, 0.0902065966656726, -0.0357037896469376, 0.0278324049067612, 0.0, 0.0, 0.0]
    fist_values = [0.0458526557352116, -0.0298337478538192, 0.0246731254761714, 0.0446843900924132, 0.0042330131267378, 0.0106300572598414, 0.0595302283919302, -0.0217476375326034, 0.0594795792791208, 0.0951993450696842, -0.061488174074505, 0.0199861148369101, 0.0296073461972217, -0.0204350007288065, 0.0503767142093957, 0.0501391101175333, -0.0489082549987914, 0.0011461659042025, 0.0782608738599793, -0.0200093073219586, -0.0102808406425453, 0.0994450672091517, -0.0325026184242962, 0.0222151589401808, 0.0741840370896926, 0.0, 0.0, 0.0271067407621199, -0.0143678584577016, 0.0366918077039313, 0.0054322922285664, 0.0022638132212339, 0.0189609138534228, 0.0564340331184984, -0.0350753228043586, 0.0371462921667648, 0.0351360414370634, -0.0204723423436768, 0.0278040666875756, 0.0179092878052087, -0.0094739157188277, 0.0013472776889526, 0.0442825105119198, -0.0267764221066135, 0.0429362773733665, 0.082003337335811, -0.0252419394751093, 0.0457830826274344, 0.0167094416137036, -0.0200432012792713, -0.0005725045540849, 0.0640334888667485, -0.0412945569202497, 0.027680172074905, 0.0290983128549515, 0.0081051883131955, 0.0648233341509051, 0.016204587722823, -0.002712994706759, 0.0062556031359439, -0.0068689868563961, -0.0018144176311048, 0.0237851996070874, 0.0796760401839296, -0.0568472482125976, 0.0115587974650715, 0.0794627622777087, -0.0420251604215324, 0.0338817134478528, 0.0610696592679479, 0.0104849391457576, 0.0312232458468142, 0.0375021707198111, -0.016340321699864, 0.0675526542784367, 0.0, 0.0, 0.0]
    point_values = [0.0385657492755154, -0.0357325779028907, 0.0266482465227927, 0.0345106491234739, 0.0022305976526413, 0.0293611285765446, 0.0426275700584396, -0.0345583584439747, 0.0656946494868139, 0.0748675166345342, -0.062184522396478, 0.0489916790134641, 0.017747293382934, -0.0326484557612796, 0.0505988884111973, 0.0464884986330049, -0.0452439842974745, 0.0096349880083424, 0.0747717543730375, -0.0195475623417801, -0.0202604684826312, 0.0885948769082974, -0.0414216176883482, 0.0322583277659185, 0.0730462513171856, 0.0, 0.0, 0.0172264484364398, -0.0224153921700884, 0.0398169502450042, -0.0054547635486564, -0.0054435340259086, 0.0179047236737357, 0.0492720552911478, -0.0411796906789199, 0.0382442570876014, 0.0279157628403077, -0.0252257778940293, 0.0293304496740546, 0.0167488511701889, -0.0079351921746211, 0.0005737068253914, 0.0345151105597979, -0.0343866916097072, 0.0439087143356653, 0.0722174272860065, -0.0337378895587731, 0.0531632036108351, 0.0151393408426617, -0.0175309939969682, -0.001630071100004, 0.0694454417491472, -0.0723786813423907, 0.0508974266364721, -0.0126849674051193, -0.0164842468821426, 0.0680717335774633, 0.0131810956960814, -0.0029209620895192, 0.0098338204856375, -0.0099297961018725, -0.0137839058178116, 0.0174883607841359, 0.0678330855168884, -0.0570084993597713, 0.0331802537230512, 0.0773068907677933, -0.0604515996681157, 0.0468426919879498, 0.0192322897991761, 0.0009794055172661, 0.0656147570911978, 0.0166756399939315, -0.0341479295919051, 0.0700348953286323, 0.0, 0.0, 0.0]
    thumbsup_values = [0.054632296546062, 0.0079817601893241, -0.0238114492869136, 0.0485669419317104, -0.0063644778276043, -0.0030816890488491, 0.0799569443950741, -0.0169034354757686, -0.0412109533995459, 0.0788716998419164, 0.0878680895668192, -0.0033399322963301, 0.0560158779286524, -0.024447223350181, -0.0340873481171338, 0.0459528225185655, 0.0537419851192964, -0.0089935730894081, 0.0800946626344619, 0.0207193117783465, -0.0013456877064869, 0.0961307469481189, 0.0230336694084078, -0.0338733530328913, 0.0758014386904364, 0.0, 0.0, 0.0488527890197729, -0.0164612577886546, -0.0238243724100973, 0.0188386783291798, -0.010234501282711, -0.0040302417996783, 0.0603843731946288, 0.0053544673045795, -0.0401998914382362, 0.0502756177152949, -0.0041204520769917, -0.0221245398615821, 0.020883050011398, 0.0094443637180342, -0.0057321059788637, 0.0591232807125823, -0.0084961784941829, -0.0382281618698332, 0.0882400855019645, 0.001476638642659, -0.0405819302801751, 0.0191533163185245, 0.0202722682356473, -0.0089801987135309, 0.0611649588615895, 0.0192544669496196, -0.0323977417250965, 0.0644746016562382, -0.0353763870258612, -0.0143521558880845, 0.0198426940374904, 0.0010094238812264, -0.0047220140642051, 0.0168399480856229, -0.0187344532786355, -0.0057141918583598, 0.0702125298096136, 0.0704848433140997, -0.0102979339802171, 0.0741569480698663, 0.0189032266645802, -0.0427289764946534, 0.0695458905857149, -0.0175257902422573, -0.0063092489691056, 0.0714704314368894, -0.0350385203195269, -0.0336901522877676, 0.0, 0.0, 0.0]
    messages = [empty_values, palmup_values, fist_values, point_values, thumbsup_values]

    for k, m in enumerate(messages):
        mes = []
        for i, elem in enumerate(trial_cols):
            if elem.split("_")[0] not in mes:
                mes.append(elem.split("_")[0])
                mes.append(str(m[i]) + " " + str(m[i+1]) + " " + str(m[i+2]))

        mes = str(4201+k) + " 4222222 41354234 " + ' '.join(mes)

        nframe, detected_gesture, elapsed_time = get_gesture(mes, model)
        print("Detected gesture:", detected_gesture)
        print("Elapsed time:", elapsed_time, "secs")
        print("---------------------")

