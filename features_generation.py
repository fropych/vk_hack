import networkx as nx
import pandas as pd
import numpy as np


class TrainFeatures:
    def __init__(self, frame: pd.DataFrame, is_train: bool = True):
        self.ego_frame = frame
        self.graph = nx.from_pandas_edgelist(
            frame[["u", "v", "x1", "x2", "x3"]], "u", "v", ["x1", "x2", "x3"]
        )

    def jaccard(self):
        jac = nx.jaccard_coefficient(
            self.graph, zip(self.ego_frame["u"], self.ego_frame["v"])
        )
        u_array = []
        v_array = []
        coef_array = []
        for u, v, coef in jac:
            u_array.append(u)
            v_array.append(v)
            coef_array.append(coef)
        frame = pd.DataFrame(
            {"u": u_array, "v": v_array, "jaccard_coef": coef_array})
        ego_frame_jaccard = self.ego_frame.merge(
            frame, on=["u", "v"], how="left")
        return ego_frame_jaccard

    def adamic_adar(self):
        adamic = nx.adamic_adar_index(
            self.graph, zip(self.ego_frame["u"], self.ego_frame["v"])
        )
        u_array = []
        v_array = []
        coef_array = []
        for u, v, coef in adamic:
            u_array.append(u)
            v_array.append(v)
            coef_array.append(coef)
        frame = pd.DataFrame(
            {"u": u_array, "v": v_array, "adamic_adar_coef": coef_array}
        )
        ego_frame_adamic = self.ego_frame.merge(
            frame, on=["u", "v"], how="left")
        return ego_frame_adamic

    def aggregations(self):
        # не уверен, что это правильно, но исходя из вашего кода так
        new_ego_frame = self.ego_frame.copy()

        # !!! генерим фичи по группировке

        # mean по x для (u и v)
        # сомневаюсь в целесообразности считать для x3, ибо это категории # 'x1'
        for x in ["x2", "x3"]:
            new_ego_frame["u_mean_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("mean")
            new_ego_frame["v_mean_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("mean")

        # median по x для (u и v)
        # сомневаюсь в целесообразности считать для x3, ибо это категории # 'x1'
        for x in ["x2", "x3"]:
            new_ego_frame["u_median_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("median")
            new_ego_frame["v_median_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("median")

        # max по x для (u и v)
        # для x3 считать смысла нет, это категориальная фича # 'x1'
        for x in ["x2"]:
            new_ego_frame["u_max_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("max")
            new_ego_frame["v_max_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("max")

        # std по x для (u и v)
        # для x3 считать смысла нет, это категориальная фича # 'x1'
        for x in ["x2"]:
            new_ego_frame["u_std_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("std")
            new_ego_frame["v_std_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("std")

        # mean по time для (u и v)
        new_ego_frame["u_mean_time"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("mean")
        new_ego_frame["v_mean_time"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("mean")

        new_ego_frame["u_count"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("count")
        new_ego_frame["v_count"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("count")
        new_ego_frame["u_count_mean"] = new_ego_frame.groupby(["ego_id", "u"])[
            "v_count"
        ].transform("mean")
        new_ego_frame["v_count_mean"] = new_ego_frame.groupby(["ego_id", "v"])[
            "u_count"
        ].transform("mean")
        new_ego_frame["u_mean_median"] = new_ego_frame.groupby(["ego_id", "u"])[
            "v_count"
        ].transform("median")
        new_ego_frame["v_mean_median"] = new_ego_frame.groupby(["ego_id", "v"])[
            "u_count"
        ].transform("median")
        new_ego_frame["u_mean_std"] = new_ego_frame.groupby(["ego_id", "u"])[
            "v_count"
        ].transform("std")
        new_ego_frame["v_mean_std"] = new_ego_frame.groupby(["ego_id", "v"])[
            "u_count"
        ].transform("std")
        new_ego_frame["u_mean_max"] = new_ego_frame.groupby(["ego_id", "u"])[
            "v_count"
        ].transform("max")
        new_ego_frame["v_mean_max"] = new_ego_frame.groupby(["ego_id", "v"])[
            "u_count"
        ].transform("max")
        # median по time для (u и v)
        new_ego_frame["u_median_time"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("median")
        new_ego_frame["v_median_time"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("median")

        # std по time для (u и v)
        # сомневаюсь в целом в целесообразности
        new_ego_frame["u_std_time"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("std")
        new_ego_frame["v_std_time"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("std")

        # !!! генерим фичи по сэмплам

        # отношение x / time
        # для x1 считать нельзя, это таргет
        # для x3 считать смысла нет, это категориалка
        new_ego_frame["u_" + "x2" +
                      "\time"] = new_ego_frame["x2"] / new_ego_frame["t"]

        new_ego_frame["ego_mean_count_v"] = new_ego_frame.groupby(["ego_id"])[
            "v_count"
        ].transform("mean")
        new_ego_frame["ego_mean_count_u"] = new_ego_frame.groupby(["ego_id"])[
            "u_count"
        ].transform("mean")
        new_ego_frame["ego_mean_t"] = new_ego_frame.groupby(["ego_id"])["t"].transform(
            "mean"
        )
        new_ego_frame["ego_mean_x2"] = new_ego_frame.groupby(["ego_id"])[
            "x1"
        ].transform("mean")

        new_ego_frame["ego_max_count_v"] = new_ego_frame.groupby(["ego_id"])[
            "v_count"
        ].transform("max")
        new_ego_frame["ego_max_count_u"] = new_ego_frame.groupby(["ego_id"])[
            "u_count"
        ].transform("max")
        new_ego_frame["ego_max_t"] = new_ego_frame.groupby(["ego_id"])["t"].transform(
            "max"
        )
        new_ego_frame["ego_max_x2"] = new_ego_frame.groupby(["ego_id"])["x1"].transform(
            "max"
        )

        new_ego_frame["ego_median_count_v"] = new_ego_frame.groupby(["ego_id"])[
            "v_count"
        ].transform("median")
        new_ego_frame["ego_median_count_u"] = new_ego_frame.groupby(["ego_id"])[
            "u_count"
        ].transform("median")
        new_ego_frame["ego_median_t"] = new_ego_frame.groupby(["ego_id"])[
            "t"
        ].transform("median")
        new_ego_frame["ego_median_x2"] = new_ego_frame.groupby(["ego_id"])[
            "x1"
        ].transform("median")

        return new_ego_frame

    # def katz(self):
    #     centrality = nx.katz_centrality(self.graph, alpha=0.1, max_iter=100000)
    #     vertex_array = []
    #     katz_array = []
    #     for vertex, centr in sorted(centrality.items()):
    #         vertex_array.append(vertex)
    #         katz_array.append(centr)
    #     frame = pd.DataFrame({'u': vertex_array, 'katz_u': katz_array})
    #     ego_frame_adamic = self.ego_frame.merge(frame, on='u', how='left')
    #     frame = pd.DataFrame({'v': vertex_array, 'katz_v': katz_array})
    #     ego_frame_adamic = ego_frame_adamic.merge(frame, on='v', how='left')
    #     return ego_frame_adamic

    def common_neighbors(self):
        lens = self.ego_frame.apply(
            lambda row: sum(1 for i in nx.common_neighbors(
                self.graph, row.u, row.v)),
            axis=1,
        )
        return self.ego_frame.assign(common_neighbors=lens.values)

    def shortest_paths(self):
        def sh_path(graph, ego_id, node):
            try:
                return len(nx.shortest_path(graph, ego_id, node))
            except:
                return 0

        def w_sh_path(graph, ego_id, node, x):
            try:
                return len(nx.shortest_path(graph, ego_id, node, weight=x))
            except:
                return 0

        return self.ego_frame.assign(
            # sh_path_u = pd.DataFrame(self.ego_frame.apply(lambda row: sh_path(self.graph, row.ego_id, row.u), axis=1), index=self.ego_frame.index),
            # sh_path_v = pd.DataFrame(self.ego_frame.apply(lambda row: sh_path(self.graph, row.ego_id, row.v), axis=1), index=self.ego_frame.index),
            sh_path_u_x1=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.u, "x1"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            sh_path_v_x1=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.v, "x1"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            sh_path_u_x2=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.u, "x2"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            sh_path_v_x2=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.v, "x2"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            # sh_path_u_x3 = pd.DataFrame(self.ego_frame.apply(lambda row: w_sh_path(self.graph, row.ego_id, row.u, 'x3'), axis=1), index=self.ego_frame.index),
            # sh_path_v_x3 = pd.DataFrame(self.ego_frame.apply(lambda row: w_sh_path(self.graph, row.ego_id, row.v, 'x3'), axis=1), index=self.ego_frame.index),
        )

    # похожесть двух нод
    def simrank_similarity(self):
        data = self.ego_frame.apply(
            lambda row: nx.simrank_similarity(self.graph, row.u, row.v), axis=1
        )
        return self.ego_frame.assign(common_neighbors=data.values)

    # Compute the preferential attachment score of all node pairs in ebunch.
    # Примерно как жаккард
    def preferential_attachment(self):
        preferential_attachment = nx.preferential_attachment(
            self.graph, zip(self.ego_frame["u"], self.ego_frame["v"])
        )
        u_array = []
        v_array = []
        coef_array = []

        for u, v, coef in preferential_attachment:
            u_array.append(u)
            v_array.append(v)
            coef_array.append(coef)
        frame = pd.DataFrame(
            {"u": u_array, "v": v_array, "preferential_attachment": coef_array}
        )
        ego_frame_pref = self.ego_frame.merge(frame, on=["u", "v"], how="left")
        return ego_frame_pref

    def make_dataset(self):
        # jac = self.jaccard()
        # adar = self.adamic_adar().drop(columns=features) # ['t', 'x1', 'x2', 'x3']
        # aggr = self.aggregations().drop(columns=features)
        aggr = self.aggregations()
        # neighbors = self.common_neighbors().drop(columns=features)
        # # similarity = self.simrank_similarity().drop(columns=features)
        # attachment = self.preferential_attachment().drop(columns=features)
        # # paths = self.shortest_paths().drop(columns=['t', 'x1', 'x2', 'x3'])
        # jac = jac.merge(adar, on=['ego_id', 'u', 'v'], how='left')
        # jac = jac.merge(aggr, on=['ego_id', 'u', 'v'], how='left')
        # jac = jac.merge(neighbors, on=['ego_id', 'u', 'v'], how='left')
        # # jac = jac.merge(similarity, on=['ego_id', 'u', 'v'], how='left')
        # jac = jac.merge(attachment, on=['ego_id', 'u', 'v'], how='left')
        # # jac = jac.merge(paths, on=['ego_id', 'u', 'v'], how='left')
        # return jac
        return aggr


class TestFeatures:
    def __init__(self, frame: pd.DataFrame):
        self.ego_frame = frame.copy(deep=True)
        self.ego_frame["x1"].fillna(self.ego_frame["x1"].mean(), inplace=True)
        self.graph = nx.from_pandas_edgelist(
            self.ego_frame[["u", "v", "x1", "x2", "x3"]
                           ], "u", "v", ["x1", "x2", "x3"]
        )

    def jaccard(self):
        jac = nx.jaccard_coefficient(
            self.graph, zip(self.ego_frame["u"], self.ego_frame["v"])
        )
        u_array = []
        v_array = []
        coef_array = []
        for u, v, coef in jac:
            u_array.append(u)
            v_array.append(v)
            coef_array.append(coef)
        frame = pd.DataFrame(
            {"u": u_array, "v": v_array, "jaccard_coef": coef_array})
        ego_frame_jaccard = self.ego_frame.merge(
            frame, on=["u", "v"], how="left")
        return ego_frame_jaccard

    def adamic_adar(self):
        adamic = nx.adamic_adar_index(
            self.graph, zip(self.ego_frame["u"], self.ego_frame["v"])
        )
        u_array = []
        v_array = []
        coef_array = []

        for u, v, coef in adamic:
            u_array.append(u)
            v_array.append(v)
            coef_array.append(coef)
        frame = pd.DataFrame(
            {"u": u_array, "v": v_array, "adamic_adar_coef": coef_array}
        )
        ego_frame_adamic = self.ego_frame.merge(
            frame, on=["u", "v"], how="left")
        return ego_frame_adamic

    def aggregations(self):
        # не уверен, что это правильно, но исходя из вашего кода так
        new_ego_frame = self.ego_frame.copy()

        # !!! генерим фичи по группировке

        # mean по x для (u и v)
        # сомневаюсь в целесообразности считать для x3, ибо это категории # 'x1'
        for x in ["x2", "x3"]:
            new_ego_frame["u_mean_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("mean")
            new_ego_frame["v_mean_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("mean")

        # median по x для (u и v)
        # сомневаюсь в целесообразности считать для x3, ибо это категории # 'x1'
        for x in ["x2", "x3"]:
            new_ego_frame["u_median_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("median")
            new_ego_frame["v_median_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("median")

        # max по x для (u и v)
        # для x3 считать смысла нет, это категориальная фича # 'x1'
        for x in ["x2"]:
            new_ego_frame["u_max_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("max")
            new_ego_frame["v_max_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("max")

        # std по x для (u и v)
        # для x3 считать смысла нет, это категориальная фича # 'x1'
        for x in ["x2"]:
            new_ego_frame["u_std_" + x] = new_ego_frame.groupby(["ego_id", "u"])[
                x
            ].transform("std")
            new_ego_frame["v_std_" + x] = new_ego_frame.groupby(["ego_id", "v"])[
                x
            ].transform("std")

        # mean по time для (u и v)
        new_ego_frame["u_mean_time"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("mean")
        new_ego_frame["v_mean_time"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("mean")

        # median по time для (u и v)
        new_ego_frame["u_median_time"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("median")
        new_ego_frame["v_median_time"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("median")

        # std по time для (u и v)
        # сомневаюсь в целом в целесообразности
        new_ego_frame["u_std_time"] = new_ego_frame.groupby(["ego_id", "u"])[
            "t"
        ].transform("std")
        new_ego_frame["v_std_time"] = new_ego_frame.groupby(["ego_id", "v"])[
            "t"
        ].transform("std")

        # !!! генерим фичи по сэмплам

        # отношение x / time
        # для x1 считать нельзя, это таргет
        # для x3 считать смысла нет, это категориалка
        new_ego_frame["u_" + "x2" +
                      "\time"] = new_ego_frame["x2"] / new_ego_frame["t"]

        return new_ego_frame

    # def katz(self):
    #     centrality = nx.katz_centrality(self.graph, alpha=0.1, max_iter=100000)
    #     vertex_array = []
    #     katz_array = []
    #     for vertex, centr in sorted(centrality.items()):
    #         vertex_array.append(vertex)
    #         katz_array.append(centr)
    #     frame = pd.DataFrame({'u': vertex_array, 'katz_u': katz_array})
    #     ego_frame_adamic = self.ego_frame.merge(frame, on='u', how='left')
    #     frame = pd.DataFrame({'v': vertex_array, 'katz_v': katz_array})
    #     ego_frame_adamic = ego_frame_adamic.merge(frame, on='v', how='left')
    #     return ego_frame_adamic

    def common_neighbors(self):
        lens = self.ego_frame.apply(
            lambda row: sum(1 for i in nx.common_neighbors(
                self.graph, row.u, row.v)),
            axis=1,
        )
        return self.ego_frame.assign(common_neighbors=lens.values)

    def shortest_paths(self):
        def sh_path(graph, ego_id, node):
            try:
                return len(nx.shortest_path(graph, ego_id, node))
            except:
                return 0

        def w_sh_path(graph, ego_id, node, x):
            try:
                return len(nx.shortest_path(graph, ego_id, node, weight=x))
            except:
                return 0

        return self.ego_frame.assign(
            # sh_path_u = pd.DataFrame(self.ego_frame.apply(lambda row: sh_path(self.graph, row.ego_id, row.u), axis=1), index=self.ego_frame.index),
            # sh_path_v = pd.DataFrame(self.ego_frame.apply(lambda row: sh_path(self.graph, row.ego_id, row.v), axis=1), index=self.ego_frame.index),
            sh_path_u_x1=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.u, "x1"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            sh_path_v_x1=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.v, "x1"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            sh_path_u_x2=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.u, "x2"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            sh_path_v_x2=pd.DataFrame(
                self.ego_frame.apply(
                    lambda row: w_sh_path(self.graph, row.ego_id, row.v, "x2"), axis=1
                ),
                index=self.ego_frame.index,
            ),
            # sh_path_u_x3 = pd.DataFrame(self.ego_frame.apply(lambda row: w_sh_path(self.graph, row.ego_id, row.u, 'x3'), axis=1), index=self.ego_frame.index),
            # sh_path_v_x3 = pd.DataFrame(self.ego_frame.apply(lambda row: w_sh_path(self.graph, row.ego_id, row.v, 'x3'), axis=1), index=self.ego_frame.index),
        )

    # похожесть двух нод
    def simrank_similarity(self):
        data = self.ego_frame.apply(
            lambda row: nx.simrank_similarity(self.graph, row.u, row.v), axis=1
        )
        return self.ego_frame.assign(common_neighbors=data.values)

    # Compute the preferential attachment score of all node pairs in ebunch.
    # Примерно как жаккард
    def preferential_attachment(self):
        preferential_attachment = nx.preferential_attachment(
            self.graph, zip(self.ego_frame["u"], self.ego_frame["v"])
        )
        u_array = []
        v_array = []
        coef_array = []

        for u, v, coef in preferential_attachment:
            u_array.append(u)
            v_array.append(v)
            coef_array.append(coef)
        frame = pd.DataFrame(
            {"u": u_array, "v": v_array, "preferential_attachment": coef_array}
        )
        ego_frame_pref = self.ego_frame.merge(frame, on=["u", "v"], how="left")
        return ego_frame_pref

    def make_dataset(self):
        # jac = self.jaccard()
        # adar = self.adamic_adar().drop(columns=features) # ['t', 'x1', 'x2', 'x3']
        # aggr = self.aggregations().drop(columns=features)
        aggr = self.aggregations()
        # neighbors = self.common_neighbors().drop(columns=features)
        # # similarity = self.simrank_similarity().drop(columns=features)
        # attachment = self.preferential_attachment().drop(columns=features)
        # # paths = self.shortest_paths().drop(columns=['t', 'x1', 'x2', 'x3'])
        # jac = jac.merge(adar, on=['ego_id', 'u', 'v'], how='left')
        # jac = jac.merge(aggr, on=['ego_id', 'u', 'v'], how='left')
        # jac = jac.merge(neighbors, on=['ego_id', 'u', 'v'], how='left')
        # # jac = jac.merge(similarity, on=['ego_id', 'u', 'v'], how='left')
        # jac = jac.merge(attachment, on=['ego_id', 'u', 'v'], how='left')
        # # jac = jac.merge(paths, on=['ego_id', 'u', 'v'], how='left')
        # return jac
        return aggr
