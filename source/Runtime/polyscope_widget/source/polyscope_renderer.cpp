#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS

#include <utility>

#include "imgui_internal.h"
#include "polyscope/render/engine.h"
#include "polyscope/view.h"

#endif

#include <cstddef>
#include <thread>
#include <vector>

#include "RHI/rhi.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "imgui.h"
#include "nvrhi/nvrhi.h"
#include "polyscope/curve_network.h"
#include "polyscope/options.h"
#include "polyscope/pick.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/screenshot.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/transformation_gizmo.h"
#include "polyscope_widget/polyscope_renderer.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usdGeom/curves.h"
#include "pxr/usd/usdGeom/points.h"
#include "pxr/usd/usdGeom/xform.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

// [0]: left Ctrl + left mouse button, [1]: left mouse button, [2]: middle mouse
// button
std::vector<std::pair<polyscope::Structure*, size_t>>
    PolyscopeRenderer::pick_result = { { nullptr, 0 },
                                       { nullptr, 0 },
                                       { nullptr, 0 } };

// int nPts = 2000;
// float anotherParam = 0.0;

// void testSubroutine()
// {
//     // do something useful...

//     // Register a structure
//     std::vector<glm::vec3> points;
//     for (int i = 0; i < nPts; i++) {
//         points.push_back(glm::vec3{ polyscope::randomUnit(),
//                                     polyscope::randomUnit(),
//                                     polyscope::randomUnit() });
//     }
//     polyscope::registerPointCloud("my point cloud", points);
// }

// Your callback functions
// void testCallback()
// {
//     // Since options::openImGuiWindowForUserCallback == true by default,
//     // we can immediately start using ImGui commands to build a UI

//     ImGui::PushItemWidth(100);  // Make ui elements 100 pixels wide,
//                                 // instead of full width. Must have
//                                 // matching PopItemWidth() below.

//     ImGui::InputInt("num points", &nPts);             // set a int variable
//     ImGui::InputFloat("param value", &anotherParam);  // set a float variable

//     if (ImGui::Button("run subroutine")) {
//         // executes when button is pressed
//         testSubroutine();
//     }
//     ImGui::SameLine();
//     if (ImGui::Button("hi")) {
//         polyscope::warning("hi");
//     }

//     ImGui::PopItemWidth();
// }

PolyscopeRenderer::PolyscopeRenderer(Stage* stage)
    : stage_(stage),
      stage_listener(stage_)
{
    // polyscope::options::buildGui = false;
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::init();
    polyscope::view::bgColor = { 1.0, 1.0, 1.0, 1.0 };
    // Test register a structure
    // std::vector<glm::vec3> points;
    // for (int i = 0; i < 2000; i++) {
    //     points.push_back(glm::vec3{ polyscope::randomUnit(),
    //                                 polyscope::randomUnit(),
    //                                 polyscope::randomUnit() });
    // }
    // polyscope::registerPointCloud("my point cloud", points);
    // polyscope::state::userCallback = testCallback;
    xform_cache = pxr::UsdGeomXformCache(pxr::UsdTimeCode::Default());

    for (const auto& prim : stage_->get_usd_stage()->Traverse()) {
        dirty_paths.insert(prim.GetPath());
    }
    UpdateStructures(dirty_paths);
}

PolyscopeRenderer::~PolyscopeRenderer()
{
    // Deconstruct slice planes here
    while (!polyscope::state::slicePlanes.empty()) {
        polyscope::state::slicePlanes.pop_back();
    }

    // The deconstruction is complex, I only deconstruct slice planes here to
    // avoid exceptions.
    // The polyscope::shutdown() function does not deconstruct global variables

    polyscope::shutdown();
}

bool PolyscopeRenderer::BuildUI()
{
    if (size_changed) {
        auto size = ImGui::GetContentRegionAvail();
        polyscope::view::setWindowSize(size.x, size.y);
        buffer.resize(size.x * size.y * 4);
    }

    if (buffer.size() == 0) {
        return false;
    }

    DrawFrame();
    if (enable_input_events) {
        ProcessInputEvents();
    }
    polyscope::view::updateFlight();
    // polyscope::buildUserGuiAndInvokeCallback();

    if (polyscope::options::maxFPS != -1) {
        auto currTime = std::chrono::steady_clock::now();
        long microsecPerLoop = 1000000 / polyscope::options::maxFPS;
        microsecPerLoop =
            (95 * microsecPerLoop) /
            100;  // give a little slack so we actually hit target fps
        while (std::chrono::duration_cast<std::chrono::microseconds>(
                   currTime - lastMainLoopIterTime)
                   .count() < microsecPerLoop) {
            std::this_thread::yield();
            currTime = std::chrono::steady_clock::now();
        }
    }
    lastMainLoopIterTime = std::chrono::steady_clock::now();

    return true;
}

std::string PolyscopeRenderer::GetChildWindowName()
{
    return child_window_name;
}

void PolyscopeRenderer::Set2dMode()
{
    // Set the navigation style to 2D
    polyscope::view::setNavigateStyle(polyscope::NavigateStyle::Planar);
    // Set the projection mode to orthographic
    polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
    // Set the view to the XY plane
    polyscope::view::setUpDir(polyscope::view::UpDir::YUp);
    polyscope::view::setFrontDir(polyscope::FrontDir::NegZFront);
    // Reset the camera view to the home view
    polyscope::view::resetCameraToHomeView();
    // Disable the input events
    enable_input_events = false;
}

ImGuiWindowFlags PolyscopeRenderer::GetWindowFlag()
{
    return ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoCollapse |
           ImGuiWindowFlags_NoScrollbar;
}

const char* PolyscopeRenderer::GetWindowName()
{
    return "Polyscope Renderer";
}

std::string PolyscopeRenderer::GetWindowUniqueName()
{
    return "Polyscope Renderer";
}

void PolyscopeRenderer::BackBufferResized(
    unsigned width,
    unsigned height,
    unsigned sampleCount)
{
    IWidget::BackBufferResized(width, height, sampleCount);
}

void PolyscopeRenderer::GetFrameBuffer()
{
    // buffer = polyscope::screenshotToBufferCustom(false);
    polyscope::drawCustom();
    polyscope::render::engine->swapDisplayBuffers();
    buffer = polyscope::render::engine->readDisplayBuffer();
}

void PolyscopeRenderer::DrawMenuBar()
{
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save Image")) {
                polyscope::screenshot();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}

void PolyscopeRenderer::RegisterGeometryFromPrim(const pxr::UsdPrim& prim)
{
    // Register structures from stage
    if (!prim) {
        return;
    }

    auto xform = xform_cache.GetLocalToWorldTransform(prim);
    auto primTypeName = prim.GetTypeName().GetString();

    // If the prim already exists, the type of the prim may have changed
    // Remove the existing prim and re-register it
    polyscope::removeStructure(prim.GetPath().GetString());

    if (primTypeName == "Mesh") {
        auto mesh = pxr::UsdGeomMesh(prim);

        pxr::VtArray<pxr::GfVec3f> points;
        mesh.GetPointsAttr().Get(&points);
        // std::vector<glm::vec3> vertices;
        // vertices.reserve(points.size());
        // for (const auto& point : points) {
        //     vertices.emplace_back(glm::make_vec3(point.GetArray()));
        // }

        pxr::VtArray<int> faceVertexCounts, faceVertexIndices;
        mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
        mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);
        // Nested list of faces
        std::vector<std::vector<size_t>> faceVertexIndicesNested;
        faceVertexIndicesNested.reserve(faceVertexCounts.size());
        size_t start = 0;
        for (int count : faceVertexCounts) {
            std::vector<size_t> face;
            face.reserve(count);
            for (int j = 0; j < count; ++j) {
                face.push_back(faceVertexIndices[start + j]);
            }
            faceVertexIndicesNested.push_back(std::move(face));
            start += count;
        }
        auto surface_mesh = polyscope::registerSurfaceMesh(
            prim.GetPath().GetString(),
            std::move(points),
            std::move(faceVertexIndicesNested));
        surface_mesh->setTransform(glm::make_mat4(xform.GetArray()));
    }
    else if (primTypeName == "Points") {
        auto points = pxr::UsdGeomPoints(prim);
        pxr::VtArray<pxr::GfVec3f> positions;
        points.GetPointsAttr().Get(&positions);
        auto point_cloud = polyscope::registerPointCloud(
            prim.GetPath().GetString(), std::move(positions));
        point_cloud->setTransform(glm::make_mat4(xform.GetArray()));
    }
    else if (primTypeName == "BasisCurves") {
        auto curves = pxr::UsdGeomCurves(prim);
        pxr::VtArray<pxr::GfVec3f> points;
        curves.GetPointsAttr().Get(&points);
        pxr::VtArray<int> curveVertexCounts;
        curves.GetCurveVertexCountsAttr().Get(&curveVertexCounts);

        size_t nEdges = 0;
        for (int count : curveVertexCounts) {
            nEdges += count - 1;
        }

        std::vector<std::array<size_t, 2>> edges;
        edges.reserve(nEdges);

        size_t start = 0;
        for (int count : curveVertexCounts) {
            for (int j = 0; j < count - 1; ++j) {
                edges.push_back({ start + j, start + j + 1 });
            }
            start += count;
        }
        auto curve_network = polyscope::registerCurveNetwork(
            prim.GetPath().GetString(), std::move(points), std::move(edges));
        curve_network->setTransform(glm::make_mat4(xform.GetArray()));
    }
    else {
        // TODO
    }
}

void PolyscopeRenderer::UpdateStructures(DirtyPathSet paths)
{
    // if (paths.size() > 0) {
    //     std::cout << "Update structures: " << std::endl;
    //     for (const auto& path : paths) {
    //         std::cout << path.GetString() << std::endl;
    //     }
    // }
    xform_cache = pxr::UsdGeomXformCache(stage_->get_current_time());

    for (const auto& path : dirty_paths) {
        pxr::UsdPrim prim = stage_->get_usd_stage()->GetPrimAtPath(path);
        if (!prim.IsValid()) {
            // Prim已删除，从渲染器移除
            polyscope::removeStructure(path.GetString());
            continue;
        }
        RegisterGeometryFromPrim(prim);
    }
}

void PolyscopeRenderer::DrawFrame()
{
    // DrawMenuBar();

    // Display some debug info
    // ImGui::Text(
    //     "Rendered window size: %d x %d",
    //     polyscope::view::windowWidth,
    //     polyscope::view::windowHeight);
    // ImGui::Text("io.WantCaptureMouse: %d", ImGui::GetIO().WantCaptureMouse);
    // ImGui::Text(
    //     "io.WantCaptureKeyboard: %d", ImGui::GetIO().WantCaptureKeyboard);
    // ImGui::Text("num widgets: %d", polyscope::state::widgets.size());

    // scene_dirty = true;
    {
        stage_listener.GetDirtyPaths(dirty_paths);
    }
    UpdateStructures(dirty_paths);
    GetFrameBuffer();

    ImVec2 imgui_frame_size =
        ImVec2(polyscope::view::windowWidth, polyscope::view::windowHeight);
    ImGui::BeginChild(
        "PolyscopeViewPort", imgui_frame_size, 0, ImGuiWindowFlags_NoMove);
    child_window_name = ImGui::GetCurrentWindow()->Name;
    ImGui::GetIO().WantCaptureMouse = false;

    auto nvrhi_texture = RHI::rhi_imgui_image(
        "PolyscopeViewPort",
        buffer,
        polyscope::view::windowWidth,
        polyscope::view::windowHeight,
        polyscope::view::windowWidth,
        polyscope::view::windowHeight,
        nvrhi::Format::RGBA8_UNORM);

    ImGui::Image(
        static_cast<ImTextureID>(nvrhi_texture),
        ImVec2(polyscope::view::windowWidth, polyscope::view::windowHeight),
        ImVec2(0, 1),
        ImVec2(1, 0));

    is_active = ImGui::IsWindowFocused();
    is_hovered = ImGui::IsItemHovered();
    ImGui::GetIO().WantCaptureMouse = true;
    ImGui::EndChild();
}

// 当选中顶点时，生成一个不显示的pointcloud，并显示transformation
// gizmo，用于控制顶点的位置
void PolyscopeRenderer::VisualizePickVertexGizmo(
    std::pair<polyscope::Structure*, size_t> pickResult)
{
    // 若选中的东西和curr_visualization_structure相同，则不做任何操作
    // if (curr_visualization_structure != nullptr &&
    //     pickResult.first == curr_visualization_structure) {
    //     return;
    // }

    // 若选中的东西为空，则删除当前的polyscope::structure
    if (pickResult.first == nullptr) {
        if (curr_visualization_structure != nullptr) {
            curr_visualization_structure->remove();
            curr_visualization_structure = nullptr;
        }
    }
    else {
        // 若选中的东西不为空，则创建一个polyscope::structure
        if (curr_visualization_structure != nullptr) {
            if (curr_visualization_structure == pickResult.first) {
                return;
            }
            curr_visualization_structure->remove();
            curr_visualization_structure = nullptr;
        }
        // 得到选中的东西的类型
        auto type = pickResult.first->typeName();
        auto transform = pickResult.first->getTransform();
        if (type == "Surface Mesh") {
            // 检查选中的是顶点、面、边、半边还是角
            auto surface_mesh =
                dynamic_cast<polyscope::SurfaceMesh*>(pickResult.first);
            auto ind = pickResult.second;
            // 仅当选中的是顶点时，才创建一个点云
            if (ind < surface_mesh->nVertices()) {
                // 获取顶点坐标
                auto pos = surface_mesh->vertexPositions.getValue(ind);
                // 根据顶点坐标创建一个transform
                transform = glm::translate(transform, pos);
                // 创建一个点云
                std::vector<glm::vec3> points;
                points.push_back({ 0, 0, 0 });
                curr_visualization_structure =
                    polyscope::registerPointCloud("picked point", points)
                        ->setEnabled(false)
                        ->setTransformationGizmoEnabled(true);
            }
        }
        else if (type == "Point Cloud") {
            auto point_cloud =
                dynamic_cast<polyscope::PointCloud*>(pickResult.first);
            auto ind = pickResult.second;
            auto pos = point_cloud->getPointPosition(ind);
            std::vector<glm::vec3> points;
            points.push_back(pos);
            curr_visualization_structure =
                polyscope::registerPointCloud("picked point", points)
                    ->setEnabled(false)
                    ->setTransformationGizmoEnabled(true);
        }
        else if (type == "Curve Network") {
            auto curve_network =
                dynamic_cast<polyscope::CurveNetwork*>(pickResult.first);
            auto ind = pickResult.second;
            if (ind < curve_network->nNodes()) {
                auto pos = curve_network->nodePositions.getValue(ind);
                std::vector<glm::vec3> points;
                points.push_back(pos);
                curr_visualization_structure =
                    polyscope::registerPointCloud("picked point", points)
                        ->setEnabled(false)
                        ->setTransformationGizmoEnabled(true);
            }
        }
        else {
            // TODO
        }
        if (curr_visualization_structure != nullptr) {
            curr_visualization_structure->setTransform(transform);
        }
    }
}

void PolyscopeRenderer::UpdatePickStructure(
    std::pair<polyscope::Structure*, size_t> pickResult)
{
    if (pickResult.first == nullptr ||
        curr_visualization_structure == nullptr) {
        return;
    }
    auto type = pickResult.first->typeName();
    if (type == "Surface Mesh") {
        auto surface_mesh =
            dynamic_cast<polyscope::SurfaceMesh*>(pickResult.first);
        auto ind = pickResult.second;
        // 当选中的是顶点时，用点云的transform更新顶点的位置
        if (ind < surface_mesh->nVertices()) {
            // 获取原始网格和点云的变换矩阵
            glm::mat4 meshTransform = pickResult.first->getTransform();  // T1
            glm::mat4 pointTransform =
                curr_visualization_structure->getTransform();  // T2'

            // 新顶点位置 = T1^(-1) * T2' * {0,0,0,1}
            glm::mat4 invMeshTransform = glm::inverse(meshTransform);
            glm::vec4 newPos = invMeshTransform * pointTransform *
                               glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

            // 更新网格顶点位置
            auto vertex_pos = surface_mesh->vertexPositions.data;
            vertex_pos[ind] = newPos;
            surface_mesh->updateVertexPositions(vertex_pos);
        }
    }
}

// Rewritten from processInputEvents() in polyscope.cpp
void PolyscopeRenderer::ProcessInputEvents()
{
    input_transform_triggered = false;
    input_pick_triggered = false;

    ImGuiIO& io = ImGui::GetIO();

    // If any mouse button is pressed, trigger a redraw
    if (ImGui::IsAnyMouseDown()) {
        // requestRedraw();
        polyscope::requestRedraw();
    }

    ImGuiWindow* window = ImGui::FindWindowByName(child_window_name.c_str());
    glm::vec2 windowPos{ 0., 0. };
    if (window) {
        ImVec2 p = window->Pos;
        windowPos = glm::vec2{ p.x, p.y };
    }

    // Handle transformation gizmo interactions
    bool widgetCapturedMouse = false;
    if (is_hovered) {
        for (polyscope::WeakHandle<polyscope::Widget> wHandle :
             polyscope::state::widgets) {
            if (wHandle.isValid()) {
                polyscope::Widget& w = wHandle.get();
                polyscope::TransformationGizmo* tg =
                    dynamic_cast<polyscope::TransformationGizmo*>(&w);
                if (tg) {
                    widgetCapturedMouse = tg->interactCustom(windowPos);
                    if (widgetCapturedMouse) {
                        input_transform_triggered = true;
                        if (curr_visualization_structure) {
                            UpdatePickStructure(pick_result[2]);
                        }
                        break;
                    }
                }
            }
        }
    }

    // Handle scroll events for 3D view
    if (polyscope::state::doDefaultMouseInteraction && !widgetCapturedMouse) {
        // if (!io.WantCaptureMouse && !widgetCapturedMouse) {
        if (is_active && is_hovered) {
            double xoffset = io.MouseWheelH;
            double yoffset = io.MouseWheel;

            if (xoffset != 0 || yoffset != 0) {
                polyscope::requestRedraw();

                // On some setups, shift flips the scroll direction, so take
                // the max scrolling in any direction
                double maxScroll = xoffset;
                if (std::abs(yoffset) > std::abs(xoffset)) {
                    maxScroll = yoffset;
                }

                // Pass camera commands to the camera
                if (maxScroll != 0.0) {
                    bool scrollClipPlane = io.KeyShift;

                    if (scrollClipPlane) {
                        polyscope::view::processClipPlaneShift(maxScroll);
                    }
                    else {
                        polyscope::view::processZoom(maxScroll);
                    }
                }
            }
        }

        // === Mouse inputs
        // if (!io.WantCaptureMouse && !widgetCapturedMouse) {
        if (is_hovered) {
            // Process drags
            bool dragLeft = ImGui::IsMouseDragging(0);
            bool dragRight =
                !dragLeft &&
                ImGui::IsMouseDragging(
                    1);  // left takes priority, so only one can be true
            if (dragLeft || dragRight) {
                glm::vec2 dragDelta{
                    io.MouseDelta.x / polyscope::view::windowWidth,
                    -io.MouseDelta.y / polyscope::view::windowHeight
                };
                drag_distSince_last_release += std::abs(dragDelta.x);
                drag_distSince_last_release += std::abs(dragDelta.y);

                // exactly one of these will be true
                bool isRotate = dragLeft && !io.KeyShift && !io.KeyCtrl;
                bool isTranslate =
                    (dragLeft && io.KeyShift && !io.KeyCtrl) || dragRight;
                bool isDragZoom = dragLeft && io.KeyShift && io.KeyCtrl;

                if (isDragZoom) {
                    polyscope::view::processZoom(dragDelta.y * 5);
                }
                if (isRotate) {
                    glm::vec2 currPos{ (io.MousePos.x - windowPos.x) /
                                           polyscope::view::windowWidth,
                                       1.0 -
                                           (io.MousePos.y - windowPos.y) /
                                               polyscope::view::windowHeight };

                    currPos = (currPos * 2.0f) - glm::vec2{ 1.0, 1.0 };
                    if (std::abs(currPos.x) <= 1.0 &&
                        std::abs(currPos.y) <= 1.0) {
                        polyscope::view::processRotate(
                            currPos - 2.0f * dragDelta, currPos);
                    }
                }
                if (isTranslate) {
                    polyscope::view::processTranslate(dragDelta);
                }
            }

            // Click picks

            // Left Ctrl + left click picks
            float dragIgnoreThreshold = 0.01;
            if (io.KeyCtrl && ImGui::IsMouseReleased(0)) {
                // Don't pick at the end of a long drag
                if (drag_distSince_last_release < dragIgnoreThreshold) {
                    // ImVec2 p = ImGui::GetMousePos();
                    ImVec2 p = ImGui::GetMousePos() - window->Pos;
                    std::pair<polyscope::Structure*, size_t> pickResult =
                        polyscope::pick::pickAtScreenCoords(
                            glm::vec2{ p.x, p.y });
                    if (pickResult.first != pick_result[0].first ||
                        pickResult.second != pick_result[0].second) {
                        input_pick_triggered = true;
                    }
                    polyscope::pick::setSelection(pickResult);
                    pick_result[0] = pickResult;
                }

                // Reset the drag distance after any release
                drag_distSince_last_release = 0.0;
            }
            // Left click picks
            else if (ImGui::IsMouseReleased(0)) {
                // Don't pick at the end of a long drag
                if (drag_distSince_last_release < dragIgnoreThreshold) {
                    // ImVec2 p = ImGui::GetMousePos();
                    ImVec2 p = ImGui::GetMousePos() - window->Pos;
                    std::pair<polyscope::Structure*, size_t> pickResult =
                        polyscope::pick::pickAtScreenCoords(
                            glm::vec2{ p.x, p.y });
                    if (pickResult.first != pick_result[1].first ||
                        pickResult.second != pick_result[1].second) {
                        input_pick_triggered = true;
                    }
                    polyscope::pick::setSelection(pickResult);
                    pick_result[1] = pickResult;
                }

                // Reset the drag distance after any release
                drag_distSince_last_release = 0.0;
            }
            else if (ImGui::IsMouseReleased(2)) {
                // Don't pick at the end of a long drag
                if (drag_distSince_last_release < dragIgnoreThreshold) {
                    // ImVec2 p = ImGui::GetMousePos();
                    ImVec2 p = ImGui::GetMousePos() - window->Pos;
                    std::pair<polyscope::Structure*, size_t> pickResult =
                        polyscope::pick::pickAtScreenCoords(
                            glm::vec2{ p.x, p.y });
                    if (pickResult.first != pick_result[2].first ||
                        pickResult.second != pick_result[2].second) {
                        input_pick_triggered = true;
                    }
                    pick_result[2] = pickResult;
                    VisualizePickVertexGizmo(pickResult);
                }

                // Reset the drag distance after any release
                drag_distSince_last_release = 0.0;
            }
            // Clear pick
            if (ImGui::IsMouseReleased(1)) {
                if (drag_distSince_last_release < dragIgnoreThreshold) {
                    polyscope::pick::resetSelection();
                    pick_result[0] = { nullptr, 0 };
                    pick_result[1] = { nullptr, 0 };
                    pick_result[2] = { nullptr, 0 };
                    if (curr_visualization_structure != nullptr) {
                        curr_visualization_structure->remove();
                        curr_visualization_structure = nullptr;
                    }
                }
                drag_distSince_last_release = 0.0;
            }
        }
    }

    // === Key-press inputs
    if (is_active) {
        polyscope::view::processKeyboardNavigation(io);
    }
}

// bool PolyscopeRenderer::JoystickButtonUpdate(int button, bool pressed)
// {
// }

// bool PolyscopeRenderer::JoystickAxisUpdate(int axis, float value)
// {
// }

// bool PolyscopeRenderer::KeyboardUpdate(
//     int key,
//     int scancode,
//     int action,
//     int mods)
// {
// }

// bool PolyscopeRenderer::MousePosUpdate(double xpos, double ypos)
// {
// }

// bool PolyscopeRenderer::MouseScrollUpdate(double xoffset, double yoffset)
// {
// }

// bool PolyscopeRenderer::MouseButtonUpdate(int button, int action, int
// mods)
// {
// }

// void PolyscopeRenderer::Animate(float elapsed_time_seconds)
// {
// }

USTC_CG_NAMESPACE_CLOSE_SCOPE
