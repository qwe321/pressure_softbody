using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using Random = System.Random;
using Unity.Burst;

namespace PressureBody
{
    public struct Triangle
    {
        public int a;
        public int b;
        public int c;
    }

    public struct Spring
    {
        public int a;
        public int b;
        public float distance;
    }

    [BurstCompile]
    public struct ResetJob : IJobParallelFor
    {
        public NativeArray<Vector3> normals;
        public NativeArray<Vector3> forces;

        public float gravity;

        public void Execute(int index)
        {
            normals[index] = Vector3.zero;
            forces[index] = Vector3.down * gravity;
        }
    }
    
    [BurstCompile]
    public struct TriangleCrossJob : IJobFor
    {
        [ReadOnly] public NativeArray<Vector3> positions;
        [ReadOnly] public NativeArray<Triangle> triangles;
        
        public NativeArray<Vector3> normals;
        public NativeArray<float> volume;
        
        public void Execute(int i)
        {
            var pa = positions[triangles[i].a];
            var pb = positions[triangles[i].b];
            var pc = positions[triangles[i].c];


            var v321 = pc.x * pb.y * pa.z;
            var v231 = pb.x * pc.y * pa.z;
            var v312 = pc.x * pa.y * pb.z;
            var v132 = pa.x * pc.y * pb.z;
            var v213 = pb.x * pa.y * pc.z;
            var v123 = pa.x * pb.y * pc.z;
            volume[0] += (-v321 + v231 + v312 - v132 - v213 + v123) / 6.0f;
            
            var cross = Vector3.Cross(pb - pa, pc - pa);

            normals[triangles[i].a] += cross;
            normals[triangles[i].b] += cross;
            normals[triangles[i].c] += cross;
        }
    }
    
    [BurstCompile]
    public struct SpringPass : IJobFor
    {
        [ReadOnly] public NativeArray<Spring> springs;
        [ReadOnly] public NativeArray<Vector3> positions;
        [ReadOnly] public NativeArray<Vector3> velocities;

        public NativeArray<Vector3> forces;

        public float strength;
        public float damping;
        
        public void Execute(int i)
        {
            var pa = positions[springs[i].a];
            var pb = positions[springs[i].b];
            var va = velocities[springs[i].a];
            var vb = velocities[springs[i].b];
            
            var nab = pb - pa;
            var vab = vb - va;
            var m = nab.magnitude;
            var f = (m - springs[i].distance) * strength + Vector3.Dot(vab, nab) / m * damping;
            var force = nab.normalized * f;

            forces[springs[i].a] += force;
            forces[springs[i].b] -= force;
        }
    }
    
    [BurstCompile]
    public struct PressurePass : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Vector3> normals;
        [ReadOnly] public NativeArray<Vector3> forces;

        public NativeArray<Vector3> positions;
        public NativeArray<Vector3> velocities;

        public float pressure;
        public float deltaTime;

        public void Execute(int i)
        {
            var force = forces[i];
            force += pressure * normals[i].normalized * normals[i].magnitude;

            velocities[i] += force * deltaTime;
            positions[i] = positions[i] + velocities[i] * deltaTime;
        }
    }

    
    public class PressureBodySolver : MonoBehaviour
    {
        PressureBodySolverSettings settings;

        class PinnedVertex
        {
            public int index;
            public PressureBodyPin handle;
        }

        List<PinnedVertex> pins = new List<PinnedVertex>();
        
        class Socket
        {
            public int triangle;
            public PressureBodySocket socket;
        }

        List<Socket> sockets = new List<Socket>();
        
        private int[] vertexRemapping;
 
        
        private NativeArray<Vector3> positions;
        private NativeArray<Vector3> velocities;
        private NativeArray<Vector3> normals;
        private NativeArray<Vector3> forces;
        private NativeArray<Triangle> triangles;
        private NativeArray<Spring> springs;
        private NativeArray<float> volume;
        private Vector3[] meshVertices;
        private Vector3[] meshNormals;

        private Mesh mesh;
        
        void Start()
        {
            settings = GetComponent<PressureBodySolverSettings>();
            var meshFilter = GetComponent<MeshFilter>();
            var originalMesh = meshFilter.sharedMesh;
            
            mesh = new Mesh();
            
            meshVertices = originalMesh.vertices;
            mesh.vertices = meshVertices;

            vertexRemapping = new int[meshVertices.Length];
            var uniquePositions = new List<Vector3>();
            var uniquePositionsMap = new Dictionary<Vector3, int>();
            for (int i = 0; i < meshVertices.Length; ++i)
            {
                if (uniquePositionsMap.ContainsKey(meshVertices[i]))
                {
                    vertexRemapping[i] = uniquePositionsMap[meshVertices[i]];
                }
                else
                {
                    vertexRemapping[i] = uniquePositions.Count;
                    uniquePositionsMap[meshVertices[i]] = uniquePositions.Count;
                    uniquePositions.Add(meshVertices[i]);
                }
            }
            
            meshNormals = originalMesh.normals;
            mesh.normals = meshNormals;
            mesh.uv = originalMesh.uv;
            
            var meshTriangles = originalMesh.triangles;
            mesh.triangles = originalMesh.triangles;
            
            positions = new NativeArray<Vector3>(uniquePositions.Count, Allocator.Persistent);
            velocities = new NativeArray<Vector3>(uniquePositions.Count, Allocator.Persistent);
            normals = new NativeArray<Vector3>(uniquePositions.Count, Allocator.Persistent);
            forces = new NativeArray<Vector3>(uniquePositions.Count, Allocator.Persistent);
            
            for (int i = 0; i < positions.Length; ++i)
            {
                positions[i] = uniquePositions[i];
            }

            triangles = new NativeArray<Triangle>(meshTriangles.Length / 3, Allocator.Persistent);

            var edgeMap = new Dictionary<int, List<int>>();
            var springsList = new List<Spring>();

            void AddEdge(int a, int b)
            {
                if (b < a)
                {
                    a ^= b;
                    b ^= a;
                    a ^= b;
                }

                if (!edgeMap.ContainsKey(a))
                {
                    edgeMap[a] = new List<int>();
                }

                if (!edgeMap[a].Contains(b))
                {
                    edgeMap[a].Add(b);
                    springsList.Add(new Spring()
                    {
                        a = a, b = b,
                        distance = Vector3.Distance(uniquePositions[a], uniquePositions[b])
                    });
                }
            }
            
            for (int i = 0; i < meshTriangles.Length; i += 3)
            {
                var t = new Triangle()
                {
                    a = vertexRemapping[meshTriangles[i + 0]],
                    b = vertexRemapping[meshTriangles[i + 1]],
                    c = vertexRemapping[meshTriangles[i + 2]],
                };
                
                triangles[i / 3] = t;
                
                AddEdge(t.a, t.b);
                AddEdge(t.b, t.c);
                AddEdge(t.c, t.a);
            }

            springs = new NativeArray<Spring>(springsList.Count, Allocator.Persistent);
            for (int i = 0; i < springsList.Count; ++i)
            {
                springs[i] = springsList[i];
            }
            
            volume = new NativeArray<float>(1, Allocator.Persistent);

            meshFilter.mesh = mesh;
            
            var pinsSources = settings.pinsContainers.SelectMany(pc => pc.GetComponentsInChildren<PressureBodyPin>());
            
            foreach (var ph in pinsSources)
            {
                var best = -1;
                for (int i = 0; i < uniquePositions.Count; ++i)
                {
                    if (best == -1 ||
                        Vector3.Distance(ph.transform.position, transform.TransformPoint(uniquePositions[i])) < 
                        Vector3.Distance(ph.transform.position, transform.TransformPoint(uniquePositions[best])))
                    {
                        best = i;
                    }
                }
                
                pins.Add(new PinnedVertex()
                {
                    index = best,
                    handle = ph
                });
            }

            var socketSources =
                settings.pinsContainers.SelectMany(pc => pc.GetComponentsInChildren <PressureBodySocket>());

            foreach (var ss in socketSources)
            {
                var best = -1;
                for (int i = 0; i < uniquePositions.Count; ++i)
                {
                    if (best == -1 ||
                        Vector3.Distance(ss.transform.position, transform.TransformPoint(uniquePositions[i])) < 
                        Vector3.Distance(ss.transform.position, transform.TransformPoint(uniquePositions[best])))
                    {
                        best = i;
                    }
                }

                var tri = -1;
                for (int i = 0; i < triangles.Length; ++i)
                {
                    if (best == triangles[i].a || best == triangles[i].b || best == triangles[i].c)
                    {
                        tri = i;
                        break;
                    }
                }

                var socketParent = new GameObject(ss.gameObject.name + "_parent");
                socketParent.transform.SetParent(ss.transform.parent);
                ApplyTriangleTransformToSocket(tri, socketParent.transform);
                
                ss.transform.SetParent(socketParent.transform, true);

                sockets.Add(new Socket()
                {
                    triangle = tri,
                    socket = ss
                });
            }
            
            
            Debug.Log($"Build mesh with {uniquePositions.Count} pos, {meshTriangles.Length} tris, {springsList.Count} springs, {pins.Count} pins.");
        }
        
        void ApplyTriangleTransformToSocket(int tri, Transform trans) 
        {
            var ta = positions[triangles[tri].a];
            var tb = positions[triangles[tri].b];
            var tc = positions[triangles[tri].c];

            var normal = Vector3.Cross(tb - ta, tc - ta).normalized;
            var binormal = (tb - ta - Vector3.Project(tb - ta, normal)).normalized;
            var tangent = Vector3.Cross(normal, binormal);

            trans.position = ta;
            trans.rotation = Quaternion.LookRotation(binormal, normal);
        }

        private void OnDestroy()
        {
            positions.Dispose();
            normals.Dispose();
            triangles.Dispose();
            springs.Dispose();
            volume.Dispose();
            forces.Dispose();
            velocities.Dispose();
        }

        void Simulate()
        {
            volume[0] = 0;

            var reset = new ResetJob()
            {
                normals = normals,
                forces = forces,
                gravity = settings.gravity
            };
            
            var dep = new JobHandle();
            var resetHandle = reset.Schedule(positions.Length, 64, dep);
            resetHandle.Complete();

            var triangleCross = new TriangleCrossJob()
            {
                triangles = triangles,
                normals = normals,
                positions = positions,
                volume = volume
            };
            
            var crossHandle = triangleCross.Schedule(triangles.Length, resetHandle);
            crossHandle.Complete();
 
            var springsJob = new SpringPass()
            {
                springs = springs,
                positions = positions,
                forces = forces,
                velocities = velocities,
                strength = settings.springStrength,
                damping = settings.springDamping
            };

            var springsHandle = springsJob.Schedule(springs.Length, crossHandle);
            springsHandle.Complete();

            var pressureJob = new PressurePass()
            {
                positions = positions,
                normals = normals,
                forces = forces,
                velocities = velocities,
                pressure = settings.pressure * settings.pressureScale / volume[0],
                deltaTime = settings.timeStep,
            };

            var pressureHandle = pressureJob.Schedule(positions.Length, 64, springsHandle);
            pressureHandle.Complete();

            for (int i = 0; i < pins.Count; ++i)
            {
                velocities[pins[i].index] = Vector3.zero;
                positions[pins[i].index] = transform.InverseTransformPoint(pins[i].handle.transform.position);
            }
            
        }

        void Update()
        {
            var startTime = Time.realtimeSinceStartup;
            
            for (int i = 0; i < settings.speed; ++i)
            {
                Simulate();
            }
            
            for (int i = 0; i < meshVertices.Length; ++i)
            {
                meshVertices[i] = positions[vertexRemapping[i]];
                meshNormals[i] = normals[vertexRemapping[i]];
            }
            
            mesh.vertices = meshVertices;
            mesh.normals = meshNormals;
            mesh.RecalculateBounds();

            foreach (var s in sockets)
            {
                ApplyTriangleTransformToSocket(s.triangle, s.socket.transform);
            }
            
            //Debug.Log($"Total physics time {(Time.realtimeSinceStartup-startTime)*1000} ms");
        }
    }
        
}
