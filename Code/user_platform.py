from pybbn.graph.jointree import EvidenceBuilder
import pandas as pd

def print_probs(join_tree):
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')

def user_input(join_tree, dtc, rfc, nbc):
    questions = ['Are you suffering form cold and cough?', 'Are you suffering with fever?',
                 'are you having sore throat?', 'are you feeling difficulty while breathing?',
                 'are you having headache ?', 'is you age >60?','did you have any contact with covid positive person?'
                 ]
    usernodes = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'immunity', 'contact']

    print('answer the following questions: (y or n)')
    userinput = []
    bayeinput = []
    for n, q in zip(usernodes, questions):
        a = input(q)
        if q == 'is you age >60?' and a == 'y':
            bayeinput.append('low')
            userinput.append('1')
        elif q == 'is you age >60?' and a == 'n':
            bayeinput.append('high')
            userinput.append('0')
        elif a == 'y':
            userinput.append('1')
            bayeinput.append('1')
        elif a == 'n':
            userinput.append('0')
            bayeinput.append('0')
        else:
            usernodes.remove(n)  # doesn't include the value if not answered


    bayesian(join_tree, usernodes, bayeinput)
    userinput=pd.DataFrame ([userinput], columns = usernodes)
    #print(userinput.head())
    print('prediction from desition tree',dtc.predict(userinput))
    print('prediction from random forest',rfc.predict(userinput))
    print('prediction from naive bayes',nbc.predict(userinput))


def bayesian(join_tree,usernodes,userinput):
    result = evidence(join_tree, 'ev1', usernodes, userinput, 1)
    (a, b) = str(result).split('\n')
    percentage = (float(b.split('|')[1]) * 100)
    print('percentage of risk having covid', percentage)

    if percentage<30:
        print('you are at low risk')
    elif percentage>70:
        print('you are at high risk')
    else:
        print('you are at moderate risk')

    return result





def evidence(join_tree, ev, nodes, given_evidences, val):
    for node, given_evidence in zip(nodes, given_evidences):
        ev = EvidenceBuilder() \
            .with_node(join_tree.get_bbn_node_by_name(node)) \
            .with_evidence(given_evidence, val) \
            .build()
        join_tree.set_observation(ev)
    return join_tree.get_bbn_potential(join_tree.get_bbn_node_by_name('corona_result'))
