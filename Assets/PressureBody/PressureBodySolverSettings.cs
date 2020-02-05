using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PressureBody
{
        public class PressureBodySolverSettings : MonoBehaviour
        {
                public float springStrength = 1;
                public float springDamping = 1;
                public float pressure = 0;
                public float gravity;
                public float timeStep = 0.02f;
                public int speed = 3;
                [Range(0.001f, 1)]
                public float pressureScale = 1;
                
                public List<GameObject> pinsContainers;
        }
}
