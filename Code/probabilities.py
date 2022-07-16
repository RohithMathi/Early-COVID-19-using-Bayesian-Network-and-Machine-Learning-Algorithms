import pandas as pd
from pgmpy.models import BayesianModel
import numpy as np
import networkx as nx  # for drawing graphs
import matplotlib.pyplot as plt  # for drawing graphs
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController


def bayesiannet(df):
    cough = "cough"
    fever = "fever"
    sore_throat = "sore_throat"
    shortness_of_breath = "shortness_of_breath"
    head_ache = "head_ache"
    contact = "contact"
    immunity = "immunity"
    corona_result = "corona_result"

    independent_variables = [cough, fever, sore_throat, shortness_of_breath, head_ache, contact, immunity]

    edges_list = [(immunity, fever),
                  (immunity, cough),
                  (fever, corona_result),
                  (cough, corona_result),
                  (sore_throat, corona_result),
                  (shortness_of_breath, corona_result),
                  (head_ache, corona_result),
                  (contact, corona_result)]

    print(df.head())
    
    (prob_0, prob_1) = df[immunity].value_counts(normalize=True).sort_index(ascending=False)
    immunity1 = BbnNode(Variable(6, 'immunity', ['low', 'high']), [prob_0,prob_1])


    prob = pd.crosstab(df[immunity], df[cough],  normalize='index').sort_index(ascending=False).to_numpy().reshape(-1).tolist()
    cough1 = BbnNode(Variable(0, 'cough', ['0', '1']), prob)

    prob = pd.crosstab(df[immunity], df[fever],  normalize='index').sort_index(ascending=False).to_numpy().reshape(-1).tolist()
    fever1 = BbnNode(Variable(1, 'fever', ['0', '1']), prob)

    (prob_0, prob_1) = df[sore_throat].value_counts(normalize=True).sort_index()
    sore_throat1 = BbnNode(Variable(2, 'sore_throat', ['0', '1']), [prob_0, prob_1])

    (prob_0, prob_1) = df[shortness_of_breath].value_counts(normalize=True).sort_index()
    shortness_of_breath1 = BbnNode(Variable(3, 'shortness_of_breath', ['0', '1']), [prob_0, prob_1])

    (prob_0, prob_1) = df[head_ache].value_counts(normalize=True).sort_index()
    head_ache1 = BbnNode(Variable(4, 'head_ache', ['0', '1']), [prob_0, prob_1])

    (prob_0, prob_1) = df[contact].value_counts(normalize=True).sort_index()
    contact1 = BbnNode(Variable(5, 'contact', ['0', '1']), [prob_0, prob_1])

    prob = pd.crosstab([df[cough], df[fever], df[sore_throat], df[shortness_of_breath], df[head_ache], df[contact]],
                       df[corona_result],  normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    corona_result1 = BbnNode(Variable(7, 'corona_result', ['0', '1']), prob)

    # adding nodes and edges in graph
    bayesian_net = Bbn() \
        .add_node(cough1) \
        .add_node(fever1) \
        .add_node(sore_throat1) \
        .add_node(shortness_of_breath1) \
        .add_node(head_ache1) \
        .add_node(contact1) \
        .add_node(immunity1) \
        .add_node(corona_result1) \
        .add_edge(Edge(immunity1, cough1, EdgeType.DIRECTED)) \
        .add_edge(Edge(immunity1, fever1, EdgeType.DIRECTED)) \
        .add_edge(Edge(cough1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(fever1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(sore_throat1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(shortness_of_breath1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(head_ache1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(contact1, corona_result1, EdgeType.DIRECTED))

    join_tree = InferenceController.apply(bayesian_net)
    # Set node positions
    pos = {0: (-3, -1), 1: (-2, 0), 2: (-1, 1), 3: (0.2, 1.5), 4: (1.4, 1), 5: (2.4, 0), 6: (-3, 1.5), 7: (0, -2)}

    # Set options for graph looks
    options = {
        "font_size": 12,
        "node_size": 5050,
        "node_color": "white",
        "edgecolors": "black",
        "edge_color": "red",
        "linewidths": 5,
        "width": 4}

    # Generate graph
    n, d = bayesian_net.to_nx_graph()

    nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

    # Update margins and print the graph
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()
    return join_tree,bayesian_net


def resetbbn(df):
    cough = "cough"
    fever = "fever"
    sore_throat = "sore_throat"
    shortness_of_breath = "shortness_of_breath"
    head_ache = "head_ache"
    contact = "contact"
    immunity = "immunity"
    corona_result = "corona_result"

    independent_variables = [cough, fever, sore_throat, shortness_of_breath, head_ache, contact, immunity]

    edges_list = [(immunity, fever),
                  (immunity, cough),
                  (fever, corona_result),
                  (cough, corona_result),
                  (sore_throat, corona_result),
                  (shortness_of_breath, corona_result),
                  (head_ache, corona_result),
                  (contact, corona_result)]

    (prob_0, prob_1) = df[immunity].value_counts(normalize=True).sort_index(ascending=False)
    immunity1 = BbnNode(Variable(6, 'immunity', ['low', 'high']), [prob_0, prob_1])

    prob = pd.crosstab(df[immunity], df[cough],  normalize='index').sort_index(ascending=False).to_numpy().reshape(
        -1).tolist()
    cough1 = BbnNode(Variable(0, 'cough', ['0', '1']), prob)

    prob = pd.crosstab(df[immunity], df[fever],  normalize='index').sort_index(ascending=False).to_numpy().reshape(
        -1).tolist()
    fever1 = BbnNode(Variable(1, 'fever', ['0', '1']), prob)

    (prob_0, prob_1) = df[sore_throat].value_counts(normalize=True).sort_index()
    sore_throat1 = BbnNode(Variable(2, 'sore_throat', ['0', '1']), [prob_0, prob_1])

    (prob_0, prob_1) = df[shortness_of_breath].value_counts(normalize=True).sort_index()
    shortness_of_breath1 = BbnNode(Variable(3, 'shortness_of_breath', ['0', '1']), [prob_0, prob_1])

    (prob_0, prob_1) = df[head_ache].value_counts(normalize=True).sort_index()
    head_ache1 = BbnNode(Variable(4, 'head_ache', ['0', '1']), [prob_0, prob_1])

    (prob_0, prob_1) = df[contact].value_counts(normalize=True).sort_index()
    contact1 = BbnNode(Variable(5, 'contact', ['0', '1']), [prob_0, prob_1])

    prob = pd.crosstab([df[cough], df[fever], df[sore_throat], df[shortness_of_breath], df[head_ache], df[contact]],
                       df[corona_result],  normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    corona_result1 = BbnNode(Variable(7, 'corona_result', ['0', '1']), prob)

    # adding nodes and edges in graph
    bayesian_net = Bbn() \
        .add_node(cough1) \
        .add_node(fever1) \
        .add_node(sore_throat1) \
        .add_node(shortness_of_breath1) \
        .add_node(head_ache1) \
        .add_node(contact1) \
        .add_node(immunity1) \
        .add_node(corona_result1) \
        .add_edge(Edge(immunity1, cough1, EdgeType.DIRECTED)) \
        .add_edge(Edge(immunity1, fever1, EdgeType.DIRECTED)) \
        .add_edge(Edge(cough1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(fever1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(sore_throat1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(shortness_of_breath1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(head_ache1, corona_result1, EdgeType.DIRECTED)) \
        .add_edge(Edge(contact1, corona_result1, EdgeType.DIRECTED))

    join_tree = InferenceController.apply(bayesian_net)
    return join_tree


if __name__ == '__main__':
    dforg = pd.read_csv("corona_tested_individuals_modified.csv", encoding='utf-8')
    join_tree = bayesiannet(dforg)
