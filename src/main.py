from constants import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Edge:
    normal: np.ndarray[float] 
    points: List[np.ndarray]

@dataclass
class Cone:
    vecs: List[np.ndarray]

class Polytope:
    def __init__(self):
        self.centroid: np.ndarray = np.sum(points, axis=0) / len(points)
        self.edges: List[Edge] = self.init_edges()
        self.cones: List[Cone] = self.init_cones()
        self.cutted_cones: List[Cone] = self.cut_cones()
        self.biorthogonal_cutted_cones: List[Cone] = self.biorthogonalize()

    def init_edges(self) -> List[Edge]:
        res = []
        for edge in edges:
            p = np.array([points[e-1] for e in edge])
            res.append(Edge(None, p))
        return self.init_normals(res)
    
    def init_normals(self, edges) -> List[np.ndarray[float]]:
        for e in edges:
            p1, p2, p3 = e.points

            v1 = p2 - p1
            v1 = v1 / np.linalg.norm(v1)
            v2 = p3 - p1
            v2 = v2 / np.linalg.norm(v2)

            n = np.cross(v1, v2)
            n /= np.linalg.norm(n)
            to_centroid = self.centroid - p1
            if n @  to_centroid > 0:
                n = -n
            
            e.normal = n
        return edges
    
    def init_cones(self) -> List[Cone]:
        res = []
        for c in cone_normals:
            res.append(Cone([self.edges[i-1].normal for i in c]))
        return res

    def cut_cones(self) -> List[Cone]:
        res = []

        for c in self.cones:
            n = c.vecs

            sep_normal = np.cross(n[0], n[2])
        
            side2 = np.dot(sep_normal, n[1])
            side4 = np.dot(sep_normal, n[3])
            
            if side2 * side4 < 0:
                res.append(Cone([n[0], n[1], n[2]]))
                res.append(Cone([n[0], n[2], n[3]]))
            else:
                res.append(Cone([n[1], n[0], n[3]]))
                res.append(Cone([n[1], n[2], n[3]]))

        return res

    def biorthogonalize(self) -> List[Cone]:
        res = []

        for c in self.cutted_cones:
            v1, v2, v3 = c.vecs

            b1 = np.cross(v2, v3)
            b2 = np.cross(v1, v3)
            b3 = np.cross(v1, v2)

            b1 = b1 / np.dot(b1, v1)
            b2 = b2 / np.dot(b2, v2)
            b3 = b3 / np.dot(b3, v3)

            res.append(Cone([b1, b2, b3]))

        return res 

    def draw(self) -> None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        vertices = [e.points for e in self.edges]

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # edges
        poly = Poly3DCollection(vertices, alpha=0.3)
        poly.set_facecolor('dimgray')    
        poly.set_edgecolor('black')     
    
        ax.add_collection3d(poly)

        # vertices
        ax.scatter(xs, ys, zs, s=30, c='black')
        ax.scatter(*self.centroid, s=30, c="blue")

        # normals 
        for e in self.edges:
            center = np.sum(e.points, axis=0) / len(e.points)

            ax.quiver(*center, *e.normal, length=0.4, color="black")

        # cones
        # for cone in self.biorthogonal_cutted_cones:
        v = self.biorthogonal_cutted_cones[0].vecs
        origin = self.centroid

        scale = 4.0
        p1 = origin + v[0] * scale
        p2 = origin + v[1] * scale
        p3 = origin + v[2] * scale
            
        faces = [
            [origin, p1, p2],
            [origin, p2, p3],
            [origin, p3, p1],
            [p1, p2, p3]
            ]
            
        poly = Poly3DCollection(faces, alpha=0.5, facecolor='red', edgecolor='black')
        ax.add_collection3d(poly)

        plt.show()

def solve(figure: Polytope) -> List[Tuple]:
    res = []
    j = 0
    for i in range(0, len(figure.cutted_cones), 2):
        biorth_basis1 = figure.biorthogonal_cutted_cones[i].vecs
        biorth_basis2 = figure.biorthogonal_cutted_cones[i+1].vecs
        f_biorth1 = np.array(biorth_basis1).T @ f
        f_biorth2 = np.array(biorth_basis2).T @ f
        p = points[j]
        
        if all([k >= 0 for k in f_biorth1]) or all([k >= 0 for k in f_biorth2]):
            res.append((p, f_biorth1, f_biorth2, f@p, "MAXIMUM"))
        elif all([k <= 0 for k in f_biorth1]) or all([k <= 0 for k in f_biorth2]):
            res.append((p, f_biorth1, f_biorth2, f@p, "MINIMUM"))
        j += 1

    return res

def main() -> None:
    w = Polytope()

    w.draw()

    r = solve(w)
    print("\n".join([f"{k}" for k in r]))

if __name__ == "__main__":
    main()