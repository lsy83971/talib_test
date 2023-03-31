# -*- coding: utf-8 -*-  
import math
class ClusterNode:
    def __init__(self,center):
        self.center = center

class Hierarchical():
    def __init__(self, data):
        self.data = data
        self.dist_cache = dict()

    def eucli_distance(self, col1, col2):
        if (col1, col2) in self.dist_cache:
            return self.dist_cache[(col1, col2)]
        _dist = self.data.loc[col1, col2].mean().mean()
        self.dist_cache[(col1, col2)] = _dist
        return _dist

    def traverse(self,node):
        return node.center

    def hcluster(self, col_list, ruleD=0.8, ruleN=2):
        ruleD = float(ruleD)
        ruleN = int(float(ruleN))
        nodes=[ClusterNode(center=v) for i,v in enumerate(col_list)]
        distances = {}
        currentclustid = -1
        while len(nodes) > ruleN:
            min_dist=math.inf
            nodes_len = len(nodes)
            closest_part = None
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    d_key = (nodes[i].center, nodes[j].center)
                    if d_key not in distances:
                        distances[d_key] = self.eucli_distance(
                            nodes[i].center,
                            nodes[j].center)
                    d = distances[d_key]
                    if d < min_dist and d<= ruleD:
                        min_dist = d
                        closest_part = (i, j)
            if closest_part is None:
                break
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_center = nodes[part1].center + nodes[part2].center
            new_node = ClusterNode(center=new_center)
            currentclustid -= 1
            del nodes[part2], nodes[part1]
            nodes.append(new_node)
        self.nodes = nodes
        return [self.traverse(nodes[i]) for i in range(len(nodes))]


class col_cluster(Hierarchical):
    @staticmethod
    def from_model(m):
        o = col_cluster(1 - (m.corr ** 2))
        o.m = m
        o.entL = o.m.entL
        return o

    @staticmethod
    def from_data(data, entL):
        o = col_cluster(1 - data ** 2)
        o.entL = entL
        return o

    def sort(self, cluster_res):
        cluster_res1 = [self.entL.loc[list(i)] for i in cluster_res]
        cluster_res1.sort(key=lambda x:x.max(), reverse=True)
        return cluster_res1

    def cluster(self, cols, ruleD=0.8, ruleN=2):
        col_list = [(i, ) for i in cols]
        res = self.hcluster(col_list=col_list,
                            ruleD=ruleD,
                            ruleN=ruleN)
        return self.sort(res)




class hcl:
    def __init__(self, distance):
        self.gp = {i:(j, ) for i, j in enumerate(distance.index)}
        self.next_idx = distance.shape[0]
        
        index_map = {j[0]:i for i, j in self.gp.items()}
        
        distance = distance.copy()
        distance.r1(index_map)
        distance.index = distance.columns
        for i in distance.index:
            distance.loc[i, i] = 1
        
        self.distance = distance
        self.distance_min = distance.min()

    def merge_distance(self, i1, i2):
        self.new_point(i1, i2)
        self.delete_point(i1, i2)

    def new_point(self, i1, i2):
        pass
        
    def delete_point(self, i1, i2):
        self.distance.drop([i1, i2], axis = 0, inplace = True)
        self.distance.drop([i1, i2], axis = 1, inplace = True)
        self.distance_min.drop([i1, i2], inplace = True)
        i3 = self.next_idx
        self.distance_min.loc[i3] = self.distance[i3]. min()
        del self.gp[i1]
        del self.gp[i2]
        self.next_idx += 1
        
    def min_distance(self):
        i1 = self.distance_min.idxmin()
        i2 = self.distance[i1]. idxmin()
        v = self.distance_min.loc[i1]
        return i1, i2, v
    
    def iter(self, n = 2, min_value = 0.1):
        while True:
            if len(self.gp) <= n:
                break

            i1, i2, v = self.min_distance()
            if v > min_value:
                break

            self.merge_distance(i1, i2)
        
class hcl_min(hcl):
    def new_point(self, i1, i2):
        i3 = self.next_idx
        self.distance.loc[:, i3] = self.distance[[i1, i2]]. min(axis = 1)
        self.distance.loc[i3, :] = self.distance.loc[[i1, i2], :]. min(axis = 0)
        self.distance.loc[i3, i3] = 1
        self.gp[i3] = self.gp[i1] + self.gp[i2]
    
class hcl_mean(hcl):
    def new_point(self, i1, i2):
        i3 = self.next_idx
        i1_cnt = len(self.gp[i1])
        i2_cnt = len(self.gp[i2])
        new_distance = (self.distance[i1] * i1_cnt + self.distance[i2] * i2_cnt) / (i1_cnt + i2_cnt)
        self.distance.loc[:, i3] = new_distance
        new_distance.loc[i3] = 1
        self.distance.loc[i3, :] = new_distance
        self.gp[i3] = self.gp[i1] + self.gp[i2]

if __name__ == "test":
    col_cluster.from_model(sb).cluster(sb.corr.index.tolist())[0]
    cols = [(i, ) for i in self.corr.index]
    DISTANCE = (1-(self.corr)**2)
    h = Hierarchical(data=DISTANCE)
    sg = h.hcluster(col_list=cols, ruleD=0.95, ruleN=3)
