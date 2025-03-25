#include <pxr/base/vt/array.h>

#include <vector>
#include <cmath>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

double cotangent(const OpenMesh::Vec3f& v1, const OpenMesh::Vec3f& v2) {
    double dot = v1 | v2;  
    double cross_norm = (v1 % v2).norm();  
    return dot / (cross_norm + 1e-6);  
}

void compute_mean_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& mean_curvature)
{
    mean_curvature.resize(omesh.n_vertices());
    std::fill(mean_curvature.begin(), mean_curvature.end(), 0.0f);

    for (auto v_it = omesh.vertices_begin(); v_it != omesh.vertices_end(); ++v_it) {
        MyMesh::VertexHandle vh = *v_it;
        OpenMesh::Vec3f pi = omesh.point(vh);
        OpenMesh::Vec3f laplace(0.0f, 0.0f, 0.0f);
        double area = 0.0;

        for (auto vv_it = omesh.cvv_iter(vh); vv_it.is_valid(); ++vv_it) {
            MyMesh::VertexHandle vj = *vv_it;
            OpenMesh::Vec3f pj = omesh.point(vj);
            double weight = 0.0;

            for (auto heh = omesh.cvoh_iter(vh); heh.is_valid(); ++heh) {
                if (omesh.to_vertex_handle(*heh) == vj) {
                    MyMesh::HalfedgeHandle he = *heh;
                    MyMesh::FaceHandle f1 = omesh.face_handle(he);
                    MyMesh::FaceHandle f2 = omesh.opposite_face_handle(he);

                    if (f1.is_valid() && f2.is_valid()) {
                        OpenMesh::Vec3f p1, p2;
                        std::vector<OpenMesh::Vec3f> diagonal_vertices;

                        for (auto fv_it = omesh.cfv_iter(f1); fv_it.is_valid(); ++fv_it) {
                            if (*fv_it != vh && *fv_it != vj) {
                                diagonal_vertices.push_back(omesh.point(*fv_it));
                            }
                        }
                        for (auto fv_it = omesh.cfv_iter(f2); fv_it.is_valid(); ++fv_it) {
                            if (*fv_it != vh && *fv_it != vj) {
                                diagonal_vertices.push_back(omesh.point(*fv_it));
                            }
                        }

                        if (diagonal_vertices.size() == 2) {
                            p1 = diagonal_vertices[0];
                            p2 = diagonal_vertices[1];
                            weight += 0.5 * (cotangent(p1 - pi, p1 - pj) + cotangent(p2 - pi, p2 - pj));
                        }
                    }
                }
            }
            laplace += weight * (pj - pi);
            area += weight;
        }

        if (area > 1e-6) {
            laplace /= area;
            mean_curvature[vh.idx()] = 0.5f * laplace.norm();
        }
    }
}

void compute_gaussian_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    gaussian_curvature.resize(omesh.n_vertices());
    std::fill(gaussian_curvature.begin(), gaussian_curvature.end(), 0.0f);

    for (auto v_it = omesh.vertices_begin(); v_it != omesh.vertices_end(); ++v_it) {
        MyMesh::VertexHandle vh = *v_it;
        OpenMesh::Vec3f pi = omesh.point(vh);
        double angle_sum = 0.0;
        double area = 0.0;

        for (auto vf_it = omesh.cvf_iter(vh); vf_it.is_valid(); ++vf_it) {
            MyMesh::FaceHandle fh = *vf_it;
            std::vector<OpenMesh::Vec3f> vertices;

            for (auto fv_it = omesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it) {
                if (*fv_it != vh) {
                    vertices.push_back(omesh.point(*fv_it));
                }
            }

            if (vertices.size() == 2) {
                OpenMesh::Vec3f v1 = vertices[0];
                OpenMesh::Vec3f v2 = vertices[1];

                OpenMesh::Vec3f e1 = v1 - pi;
                OpenMesh::Vec3f e2 = v2 - pi;
                e1.normalize();
                e2.normalize();
                double cos_theta = std::max(-1.0f, std::min(1.0f, e1 | e2));
                double theta = std::acos(cos_theta);
                angle_sum += theta;

                double tri_area = 0.5 * (v1 - pi).cross(v2 - pi).norm();
                area += tri_area / 3.0;  
            }
        }

        double K = (2.0 * M_PI - angle_sum) / (area + 1e-6);
        gaussian_curvature[vh.idx()] = static_cast<float>(K);
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mean_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Mean Curvature");
}

NODE_EXECUTION_FUNCTION(mean_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    // Compute mean curvature
    pxr::VtArray<float> mean_curvature;
    mean_curvature.reserve(omesh.n_vertices());

    compute_mean_curvature(omesh, mean_curvature);

    params.set_output("Mean Curvature", mean_curvature);

    return true;
}

NODE_DECLARATION_UI(mean_curvature);

NODE_DECLARATION_FUNCTION(gaussian_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Gaussian Curvature");
}

NODE_EXECUTION_FUNCTION(gaussian_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    // Compute Gaussian curvature
    pxr::VtArray<float> gaussian_curvature;
    gaussian_curvature.reserve(omesh.n_vertices());

    compute_gaussian_curvature(omesh, gaussian_curvature);

    params.set_output("Gaussian Curvature", gaussian_curvature);

    return true;
}

NODE_DECLARATION_UI(gaussian_curvature);

NODE_DEF_CLOSE_SCOPE
