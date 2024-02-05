import numpy as np
import streamlit as st
import pickle

model = pickle.load(open("Decision_Tree.pickle", 'rb'))


def get_decision_explanation(classifier, input_features):
    conversion_dict = {"odor_n": ["Odor", "None"], "stalk-root_c": ["Stalk Root", "Club"],
                       "stalk-root_r": ["Stalk Root", "Rooted"], "odor_a": ["Odor", "Almond"],
                       "odor_l": ["Odor", "Anise"], "stalk-surface_y": ["Stalk Surface", "Scaly"],
                       "spore-print-color_r": ["Spore Print Color", "Green"], "cap-surface_g": ["Cap Surface", "Grooves"],
                       "cap-shape_c": ["Cap Shape", "Conical"], "gill-size_b": ["Gill Size", "Broad"],
                       "bruises_t": ["Bruises", "Yes"], "ring-number_o": ["Ring Number", "One"]}

    columns = ['cap-shape_b', 'cap-shape_c', 'cap-shape_f', 'cap-shape_k', 'cap-shape_s', 'cap-shape_x',
               'cap-surface_f', 'cap-surface_g', 'cap-surface_s', 'cap-surface_y', 'cap-color_b', 'cap-color_c',
               'cap-color_e', 'cap-color_g', 'cap-color_n', 'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w',
               'cap-color_y', 'bruises_f', 'bruises_t', 'odor_a', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n',
               'odor_p', 'odor_s', 'odor_y', 'gill-spacing_c', 'gill-spacing_w', 'gill-size_b', 'gill-size_n',
               'gill-color_b', 'gill-color_e', 'gill-color_g', 'gill-color_h', 'gill-color_k', 'gill-color_n',
               'gill-color_o', 'gill-color_p', 'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y',
               'stalk-shape_e', 'stalk-shape_t', 'stalk-root_b', 'stalk-root_c', 'stalk-root_e', 'stalk-root_m',
               'stalk-root_r', 'stalk-surface_f', 'stalk-surface_k', 'stalk-surface_s', 'stalk-surface_y',
               'stalk-color_b', 'stalk-color_c', 'stalk-color_e', 'stalk-color_g', 'stalk-color_n', 'stalk-color_o',
               'stalk-color_p', 'stalk-color_w', 'stalk-color_y', 'ring-number_n', 'ring-number_o', 'ring-number_t',
               'ring-type_e', 'ring-type_f', 'ring-type_l', 'ring-type_n', 'ring-type_p', 'spore-print-color_b',
               'spore-print-color_h', 'spore-print-color_k', 'spore-print-color_n', 'spore-print-color_o',
               'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w', 'spore-print-color_y',
               'population_a', 'population_c', 'population_n', 'population_s', 'population_v', 'population_y',
               'habitat_d', 'habitat_g', 'habitat_l', 'habitat_m', 'habitat_p', 'habitat_u', 'habitat_w']

    # Cambiamo la forma dell'input in modo che sia compatibile con il modello
    input_features = input_features.reshape(1, -1)

    # Otteniamo il decision path e l'id dei nodi foglia
    node_indicator = classifier.decision_path(input_features)
    leaf_id = classifier.apply(input_features)

    result = []

    sample_id = 0
    # Ottiene gli id dei nodi attraversati dal campione
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]

    for node_id in node_index:
        # Se il nodo è una foglia, non lo consideriamo
        if leaf_id[sample_id] == node_id:
            continue

        # Altrimenti otteniamo il nome della feature e il valore utilizzato per la decisione
        feature_index = classifier.tree_.feature[node_id]
        feature_name = columns[feature_index] if columns else str(feature_index)

        # Aggiungiamo la spiegazione alla lista
        if input_features[sample_id, classifier.tree_.feature[node_id]] == 0:
            verb = "is not"
        else:
            verb = "is"

        explanations = "%s %s %s" % (conversion_dict[feature_name][0], verb, conversion_dict[feature_name][1])

        result.append(explanations)

    return result


mushroom_dict = {
    'cap-shape_b': 0, 'cap-shape_c': 0, 'cap-shape_f': 0, 'cap-shape_k': 0,
    'cap-shape_s': 0, 'cap-shape_x': 0, 'cap-surface_f': 0, 'cap-surface_g': 0,
    'cap-surface_s': 0, 'cap-surface_y': 0, 'cap-color_b': 0, 'cap-color_c': 0,
    'cap-color_e': 0, 'cap-color_g': 0, 'cap-color_n': 0, 'cap-color_p': 0,
    'cap-color_r': 0, 'cap-color_u': 0, 'cap-color_w': 0, 'cap-color_y': 0,
    'bruises_f': 0, 'bruises_t': 0, 'odor_a': 0, 'odor_c': 0, 'odor_f': 0,
    'odor_l': 0, 'odor_m': 0, 'odor_n': 0, 'odor_p': 0, 'odor_s': 0, 'odor_y': 0,
    'gill-spacing_c': 0, 'gill-spacing_w': 0, 'gill-size_b': 0, 'gill-size_n': 0,
    'gill-color_b': 0, 'gill-color_e': 0, 'gill-color_g': 0, 'gill-color_h': 0,
    'gill-color_k': 0, 'gill-color_n': 0, 'gill-color_o': 0, 'gill-color_p': 0,
    'gill-color_r': 0, 'gill-color_u': 0, 'gill-color_w': 0, 'gill-color_y': 0,
    'stalk-shape_e': 0, 'stalk-shape_t': 0, 'stalk-root_b': 0, 'stalk-root_c': 0,
    'stalk-root_e': 0, 'stalk_root_m': 0, 'stalk-root_r': 0, 'stalk-surface_f': 0,
    'stalk-surface_k': 0, 'stalk-surface_s': 0, 'stalk-surface_y': 0,
    'stalk-color_b': 0, 'stalk-color_c': 0, 'stalk-color_e': 0, 'stalk-color_g': 0,
    'stalk-color_n': 0, 'stalk-color_o': 0, 'stalk-color_p': 0, 'stalk-color_w': 0,
    'stalk-color_y': 0, 'ring-number_n': 0, 'ring-number_o': 0, 'ring-number_t': 0,
    'ring-type_e': 0, 'ring-type_f': 0, 'ring-type_l': 0, 'ring-type_n': 0,
    'ring-type_p': 0, 'spore-print-color_b': 0, 'spore-print-color_h': 0,
    'spore-print-color_k': 0, 'spore-print-color_n': 0, 'spore-print-color_o': 0,
    'spore-print-color_r': 0, 'spore-print-color_u': 0, 'spore-print-color_w': 0,
    'spore-print-color_y': 0, 'population_a': 0, 'population_c': 0,
    'population_n': 0, 'population_s': 0, 'population_v': 0, 'population_y': 0,
    'habitat_d': 0, 'habitat_g': 0, 'habitat_l': 0, 'habitat_m': 0,
    'habitat_p': 0, 'habitat_u': 0, 'habitat_w': 0
}

st.subheader("Cap Features")

cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)

with cap_col1:
    # Cap Shape (bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s)
    cap_shape = st.radio("Cap Shape", ["Bell", "Conical", "Convex", "Flat", "Knobbed", "Sunken"], horizontal=True)


with cap_col2:
    # Cap Surface (fibrous=f, grooves=g, scaly=y, smooth=s)
    cap_surface = st.radio("Cap Surface", ["Fibrous", "Grooves", "Scaly", "Smooth"], horizontal=True)

with cap_col3:
    # Cap Color (brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y)
    cap_color = st.radio("Cap Color", ["Brown", "Buff", "Cinnamon", "Gray", "Green", "Pink",
                                       "Purple", "Red", "White", "Yellow"], horizontal=True)

with cap_col4:
    # Bruises (bruises=t,no=f)
    bruises = st.radio("Bruises?", ["Yes", "No"], horizontal=True)


st.subheader("Odor")
# Odor (almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s)
odor = st.radio("Odor", ["Almond", "Anise", "Creosote", "Fishy", "Foul", "Musty", "None", "Pungent",
                         "Spicy"], horizontal=True)


st.subheader("Gill Features")

gill_col1, gill_col2, gill_Col3 = st.columns(3)

with gill_col1:
    # Gill Spacing (close=c,crowded=w,distant=d)
    gill_spacing = st.radio("Gill Spacing", ["Close", "Crowded"], horizontal=True)

with gill_col2:
    # Gill Size (broad=b,narrow=n)
    gill_size = st.radio("Gill Size", ["Broad", "Narrow"], horizontal=True)

with gill_Col3:
    # Gill Color (black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y)
    gill_color = st.radio("Gill Color", ["Black", "Brown", "Buff", "Chocolate", "Gray", "Green", "Orange",
                                         "Pink", "Purple", "Red", "White", "Yellow"], horizontal=True)


st.subheader("Stalk Features")

stalk_col1, stalk_col2, stalk_col3, stalk_col4 = st.columns(4)

with stalk_col1:
    # Stalk Shape (enlarging=e,tapering=t)
    stalk_shape = st.radio("Stalk Shape", ["Enlarging", "Tapering"], horizontal=True)

with stalk_col2:
    # Stalk Root (bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r)
    stalk_root = st.radio("Stalk Root", ["Bulbous", "Club", "Equal", "Rooted"], horizontal=True)

with stalk_col3:
    # Stalk Surface (fibrous=f,scaly=y,silky=k,smooth=s)
    stalk_surface = st.radio("Stalk Surface", ["Fibrous", "Scaly", "Silky", "Smooth"], horizontal=True)

with stalk_col4:
    # Stalk Color (brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y)
    stalk_color = st.radio("Stalk Color", ["Brown", "Buff", "Cinnamon", "Gray", "Orange", "Pink", "Red", "White", "Yellow"], horizontal=True)


st.subheader("Ring Features")

ring_col1, ring_col2 = st.columns(2)

with ring_col1:
    # Ring Number (none=n,one=o,two=t)
    ring_number = st.radio("Ring Number", ["None", "One", "Two"], horizontal=True)

with ring_col2:
    # Ring Type (cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z)
    ring_type = st.radio("Ring Type", ["Evanescent", "Flaring", "Large", "None", "Pendant"], horizontal=True)

st.subheader("Spore Print Color")

# Spore Print Color (black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y)
spore_print_color = st.radio("Spore Print Color", ["Black", "Brown", "Buff", "Chocolate", "Green", "Orange", "Purple", "White", "Yellow"], horizontal=True)

st.subheader("Population and Habitat")

pop_col1, pop_col2 = st.columns(2)

with pop_col1:
    # Population (abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y)
    population = st.radio("Population", ["Abundant", "Clustered", "Numerous", "Scattered", "Several", "Solitary"], horizontal=True)

with pop_col2:
    # Habitat (grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d)
    habitat = st.radio("Habitat", ["Grasses", "Leaves", "Meadows", "Paths", "Urban", "Waste", "Woods"], horizontal=True)


# Cap Shape (bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s)
if cap_shape == "Bell":
    mushroom_dict["cap-shape_b"] = 1
elif cap_shape == "Conical":
    mushroom_dict["cap-shape_c"] = 1
elif cap_shape == "Convex":
    mushroom_dict["cap-shape_x"] = 1
elif cap_shape == "Flat":
    mushroom_dict["cap-shape_f"] = 1
elif cap_shape == "Knobbed":
    mushroom_dict["cap-shape_k"] = 1
elif cap_shape == "Sunken":
    mushroom_dict["cap-shape_s"] = 1

# Cap Surface (fibrous=f,grooves=g,scaly=y,smooth=s)
if cap_surface == "Fibrous":
    mushroom_dict["cap-surface_f"] = 1
elif cap_surface == "Grooves":
    mushroom_dict["cap-surface_g"] = 1
elif cap_surface == "Scaly":
    mushroom_dict["cap-surface_y"] = 1
elif cap_surface == "Smooth":
    mushroom_dict["cap-surface_s"] = 1

# Cap Color (brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y)
if cap_color == "Brown":
    mushroom_dict["cap-color_n"] = 1
elif cap_color == "Buff":
    mushroom_dict["cap-color_b"] = 1
elif cap_color == "Cinnamon":
    mushroom_dict["cap-color_c"] = 1
elif cap_color == "Gray":
    mushroom_dict["cap-color_g"] = 1
elif cap_color == "Green":
    mushroom_dict["cap-color_r"] = 1
elif cap_color == "Pink":
    mushroom_dict["cap-color_p"] = 1
elif cap_color == "Purple":
    mushroom_dict["cap-color_u"] = 1
elif cap_color == "Red":
    mushroom_dict["cap-color_e"] = 1
elif cap_color == "White":
    mushroom_dict["cap-color_w"] = 1
elif cap_color == "Yellow":
    mushroom_dict["cap-color_y"] = 1

# Bruises (bruises=t,no=f)
if bruises == "Yes":
    mushroom_dict["bruises_t"] = 1
elif bruises == "No":
    mushroom_dict["bruises_f"] = 1

# Odor (almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s)
if odor == "Almond":
    mushroom_dict["odor_a"] = 1
elif odor == "Anise":
    mushroom_dict["odor_l"] = 1
elif odor == "Creosote":
    mushroom_dict["odor_c"] = 1
elif odor == "Fishy":
    mushroom_dict["odor_y"] = 1
elif odor == "Foul":
    mushroom_dict["odor_f"] = 1
elif odor == "Musty":
    mushroom_dict["odor_m"] = 1
elif odor == "None":
    mushroom_dict["odor_n"] = 1
elif odor == "Pungent":
    mushroom_dict["odor_p"] = 1
elif odor == "Spicy":
    mushroom_dict["odor_s"] = 1

# Gill Spacing (close=c,crowded=w,distant=d)
if gill_spacing == "Close":
    mushroom_dict["gill-spacing_c"] = 1
elif gill_spacing == "Crowded":
    mushroom_dict["gill-spacing_w"] = 1

# Gill Size (broad=b,narrow=n)
if gill_size == "Broad":
    mushroom_dict["gill-size_b"] = 1
elif gill_size == "Narrow":
    mushroom_dict["gill-size_n"] = 1

# Gill Color (black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y)
if gill_color == "Black":
    mushroom_dict["gill-color_k"] = 1
elif gill_color == "Brown":
    mushroom_dict["gill-color_n"] = 1
elif gill_color == "Buff":
    mushroom_dict["gill-color_b"] = 1
elif gill_color == "Chocolate":
    mushroom_dict["gill-color_h"] = 1
elif gill_color == "Gray":
    mushroom_dict["gill-color_g"] = 1
elif gill_color == "Green":
    mushroom_dict["gill-color_r"] = 1
elif gill_color == "Orange":
    mushroom_dict["gill-color_o"] = 1
elif gill_color == "Pink":
    mushroom_dict["gill-color_p"] = 1
elif gill_color == "Purple":
    mushroom_dict["gill-color_u"] = 1
elif gill_color == "Red":
    mushroom_dict["gill-color_e"] = 1
elif gill_color == "White":
    mushroom_dict["gill-color_w"] = 1
elif gill_color == "Yellow":
    mushroom_dict["gill-color_y"] = 1

# Stalk Shape (enlarging=e,tapering=t)
if stalk_shape == "Enlarging":
    mushroom_dict["stalk-shape_e"] = 1
elif stalk_shape == "Tapering":
    mushroom_dict["stalk-shape_t"] = 1

# Stalk Root (bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r)
if stalk_root == "Bulbous":
    mushroom_dict["stalk-root_b"] = 1
elif stalk_root == "Club":
    mushroom_dict["stalk-root_c"] = 1
elif stalk_root == "Equal":
    mushroom_dict["stalk-root_e"] = 1
elif stalk_root == "Rooted":
    mushroom_dict["stalk-root_r"] = 1

# Stalk Surface (fibrous=f,scaly=y,silky=k,smooth=s)
if stalk_surface == "Fibrous":
    mushroom_dict["stalk-surface_f"] = 1
elif stalk_surface == "Scaly":
    mushroom_dict["stalk-surface_y"] = 1
elif stalk_surface == "Silky":
    mushroom_dict["stalk-surface_k"] = 1
elif stalk_surface == "Smooth":
    mushroom_dict["stalk-surface_s"] = 1

# Stalk Color (brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y)
if stalk_color == "Brown":
    mushroom_dict["stalk-color_n"] = 1
elif stalk_color == "Buff":
    mushroom_dict["stalk-color_b"] = 1
elif stalk_color == "Cinnamon":
    mushroom_dict["stalk-color_c"] = 1
elif stalk_color == "Gray":
    mushroom_dict["stalk-color_g"] = 1
elif stalk_color == "Orange":
    mushroom_dict["stalk-color_o"] = 1
elif stalk_color == "Pink":
    mushroom_dict["stalk-color_p"] = 1
elif stalk_color == "Red":
    mushroom_dict["stalk-color_e"] = 1
elif stalk_color == "White":
    mushroom_dict["stalk-color_w"] = 1
elif stalk_color == "Yellow":
    mushroom_dict["stalk-color_y"] = 1

# Ring Number (none=n,one=o,two=t)
if ring_number == "None":
    mushroom_dict["ring-number_n"] = 1
elif ring_number == "One":
    mushroom_dict["ring-number_o"] = 1
elif ring_number == "Two":
    mushroom_dict["ring-number_t"] = 1

# Ring Type (cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z)

if ring_type == "Evanescent":
    mushroom_dict["ring-type_e"] = 1
elif ring_type == "Flaring":
    mushroom_dict["ring-type_f"] = 1
elif ring_type == "Large":
    mushroom_dict["ring-type_l"] = 1
elif ring_type == "None":
    mushroom_dict["ring-type_n"] = 1
elif ring_type == "Pendant":
    mushroom_dict["ring-type_p"] = 1


# Spore Print Color (black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y)
if spore_print_color == "Black":
    mushroom_dict["spore-print-color_k"] = 1
elif spore_print_color == "Brown":
    mushroom_dict["spore-print-color_n"] = 1
elif spore_print_color == "Buff":
    mushroom_dict["spore-print-color_b"] = 1
elif spore_print_color == "Chocolate":
    mushroom_dict["spore-print-color_h"] = 1
elif spore_print_color == "Green":
    mushroom_dict["spore-print-color_r"] = 1
elif spore_print_color == "Orange":
    mushroom_dict["spore-print-color_o"] = 1
elif spore_print_color == "Purple":
    mushroom_dict["spore-print-color_u"] = 1
elif spore_print_color == "White":
    mushroom_dict["spore-print-color_w"] = 1
elif spore_print_color == "Yellow":
    mushroom_dict["spore-print-color_y"] = 1

# Population (abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y)
if population == "Abundant":
    mushroom_dict["population_a"] = 1
elif population == "Clustered":
    mushroom_dict["population_c"] = 1
elif population == "Numerous":
    mushroom_dict["population_n"] = 1
elif population == "Scattered":
    mushroom_dict["population_s"] = 1
elif population == "Several":
    mushroom_dict["population_v"] = 1
elif population == "Solitary":
    mushroom_dict["population_y"] = 1

# Habitat (grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d)
if habitat == "Grasses":
    mushroom_dict["habitat_g"] = 1
elif habitat == "Leaves":
    mushroom_dict["habitat_l"] = 1
elif habitat == "Meadows":
    mushroom_dict["habitat_m"] = 1
elif habitat == "Paths":
    mushroom_dict["habitat_p"] = 1
elif habitat == "Urban":
    mushroom_dict["habitat_u"] = 1
elif habitat == "Waste":
    mushroom_dict["habitat_w"] = 1
elif habitat == "Woods":
    mushroom_dict["habitat_d"] = 1


if st.button("Predict"):
    result = model.predict(np.array(list(mushroom_dict.values())).reshape(1, -1))
    if result == 0:
        st.error("The mushroom is poisonous")
    else:
        st.success("The mushroom is edible")

    result = get_decision_explanation(model, np.array(list(mushroom_dict.values())))

    st.write("Why am I getting this response?")
    for i, item in enumerate(result, start=1):
        st.write(i, " - ", item)
