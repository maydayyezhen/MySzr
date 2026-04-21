using System;
using UnityEngine;

/// <summary>
/// 稳定版上肢双骨 IK 播放器
/// 说明：
/// 1. 只驱动左右上臂和前臂
/// 2. 不驱动 Hand 朝向，不驱动手指
/// 3. 使用“固定身体基底”而不是实时跟随胳膊的基底，避免反馈抖动
/// 4. 用 wrist / elbow 做简化双骨 IK
/// </summary>
public class PoseDrivePlayer : MonoBehaviour
{
    [Header("输入数据")]
    public TextAsset driveJson;

    [Header("角色组件")]
    public Animator animator;
    public Transform torsoReference;

    [Header("播放设置")]
    public float playbackSpeed = 1.0f;
    public bool loop = true;
    public bool playOnStart = true;
    [Range(1f, 30f)] public float rotationLerpSpeed = 9f;

    [Header("坐标映射设置")]
    public bool invertForward = false;
    public bool invertRight = false;
    public bool invertUp = false;
    public float positionScaleMultiplier = 0.88f;

    [Header("防穿身体约束")]
    public float bodyRadiusScale = 0.45f;
    public float wristClearanceScale = 0.16f;
    public float elbowOutwardBiasScale = 0.28f;
    public float wristForwardBiasScale = 0.08f;

    [Header("运行状态")]
    public bool isPlaying = false;
    public float currentTime = 0f;
    public int currentFrameIndex = 0;

    private DriveClipData clipData;

    private Transform leftUpperArmBone;
    private Transform leftLowerArmBone;
    private Transform leftHandBone;
    private Transform rightUpperArmBone;
    private Transform rightLowerArmBone;
    private Transform rightHandBone;
    private Transform leftHipBone;
    private Transform rightHipBone;

    private Quaternion leftUpperArmInitialLocalRotation;
    private Quaternion leftLowerArmInitialLocalRotation;
    private Quaternion rightUpperArmInitialLocalRotation;
    private Quaternion rightLowerArmInitialLocalRotation;

    private Vector3 leftUpperArmInitialAxisLocal;
    private Vector3 leftLowerArmInitialAxisLocal;
    private Vector3 rightUpperArmInitialAxisLocal;
    private Vector3 rightLowerArmInitialAxisLocal;

    private float modelShoulderWidth = 1f;
    private Vector3 initialShoulderMidWorld = Vector3.zero;
    private Vector3 initialBodyRightWorld = Vector3.right;
    private Vector3 initialBodyUpWorld = Vector3.up;
    private Vector3 initialBodyForwardWorld = Vector3.forward;

    private float leftUpperLength;
    private float leftLowerLength;
    private float rightUpperLength;
    private float rightLowerLength;

    private void Start()
    {
        if (!TryInitialize())
        {
            enabled = false;
            return;
        }

        if (playOnStart)
        {
            Play();
        }
    }

    private void Update()
    {
        if (!isPlaying || clipData == null || clipData.frames == null || clipData.frames.Length == 0)
        {
            return;
        }

        currentTime += Time.deltaTime * playbackSpeed;
        float clipLengthSec = clipData.frames.Length / Mathf.Max(clipData.fps, 1e-5f);

        if (currentTime >= clipLengthSec)
        {
            if (loop)
            {
                currentTime = 0f;
            }
            else
            {
                currentTime = clipLengthSec;
                isPlaying = false;
                return;
            }
        }

        currentFrameIndex = Mathf.Clamp(Mathf.FloorToInt(currentTime * clipData.fps), 0, clipData.frames.Length - 1);
        ApplyFrame(clipData.frames[currentFrameIndex]);
    }

    private bool TryInitialize()
    {
        if (animator == null) animator = GetComponent<Animator>();
        if (animator == null) animator = GetComponentInChildren<Animator>();
        if (animator == null)
        {
            Debug.LogError("未找到 Animator 组件。");
            return false;
        }

        if (animator.avatar == null || !animator.avatar.isHuman)
        {
            Debug.LogError("当前 Animator 不是有效的 Humanoid Avatar。");
            return false;
        }

        if (driveJson == null)
        {
            Debug.LogError("请在 Inspector 中指定 driveJson（TextAsset）。");
            return false;
        }

        clipData = JsonUtility.FromJson<DriveClipData>(driveJson.text);
        if (clipData == null || clipData.frames == null || clipData.frames.Length == 0)
        {
            Debug.LogError("drive.json 解析失败，或 frames 为空。");
            return false;
        }

        if (torsoReference == null) torsoReference = animator.GetBoneTransform(HumanBodyBones.UpperChest);
        if (torsoReference == null) torsoReference = animator.GetBoneTransform(HumanBodyBones.Chest);
        if (torsoReference == null)
        {
            Debug.LogError("未找到 UpperChest / Chest 作为 torsoReference。");
            return false;
        }

        leftUpperArmBone = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        leftLowerArmBone = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        leftHandBone = animator.GetBoneTransform(HumanBodyBones.LeftHand);
        rightUpperArmBone = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        rightLowerArmBone = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
        rightHandBone = animator.GetBoneTransform(HumanBodyBones.RightHand);
        leftHipBone = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        rightHipBone = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);

        if (leftUpperArmBone == null || leftLowerArmBone == null || leftHandBone == null ||
            rightUpperArmBone == null || rightLowerArmBone == null || rightHandBone == null)
        {
            Debug.LogError("Humanoid 手臂骨骼绑定不完整。");
            return false;
        }

        leftUpperArmInitialLocalRotation = leftUpperArmBone.localRotation;
        leftLowerArmInitialLocalRotation = leftLowerArmBone.localRotation;
        rightUpperArmInitialLocalRotation = rightUpperArmBone.localRotation;
        rightLowerArmInitialLocalRotation = rightLowerArmBone.localRotation;

        leftUpperArmInitialAxisLocal = GetInitialBoneAxisLocal(leftUpperArmBone, leftLowerArmBone);
        leftLowerArmInitialAxisLocal = GetInitialBoneAxisLocal(leftLowerArmBone, leftHandBone);
        rightUpperArmInitialAxisLocal = GetInitialBoneAxisLocal(rightUpperArmBone, rightLowerArmBone);
        rightLowerArmInitialAxisLocal = GetInitialBoneAxisLocal(rightLowerArmBone, rightHandBone);

        if (leftUpperArmInitialAxisLocal == Vector3.zero || leftLowerArmInitialAxisLocal == Vector3.zero ||
            rightUpperArmInitialAxisLocal == Vector3.zero || rightLowerArmInitialAxisLocal == Vector3.zero)
        {
            Debug.LogError("初始骨轴计算失败。");
            return false;
        }

        leftUpperLength = Vector3.Distance(leftUpperArmBone.position, leftLowerArmBone.position);
        leftLowerLength = Vector3.Distance(leftLowerArmBone.position, leftHandBone.position);
        rightUpperLength = Vector3.Distance(rightUpperArmBone.position, rightLowerArmBone.position);
        rightLowerLength = Vector3.Distance(rightLowerArmBone.position, rightHandBone.position);

        modelShoulderWidth = Vector3.Distance(leftUpperArmBone.position, rightUpperArmBone.position);
        if (modelShoulderWidth < 1e-6f) modelShoulderWidth = 1f;

        CacheInitialBodyBasis();
        return true;
    }

    private void CacheInitialBodyBasis()
    {
        initialShoulderMidWorld = (leftUpperArmBone.position + rightUpperArmBone.position) * 0.5f;

        Vector3 right = torsoReference.right.normalized;
        Vector3 up;
        if (leftHipBone != null && rightHipBone != null)
        {
            Vector3 hipMid = (leftHipBone.position + rightHipBone.position) * 0.5f;
            up = (initialShoulderMidWorld - hipMid).normalized;
        }
        else
        {
            up = torsoReference.up.normalized;
        }

        Vector3 forward = Vector3.Cross(right, up).normalized;
        up = Vector3.Cross(forward, right).normalized;

        initialBodyRightWorld = right;
        initialBodyUpWorld = up;
        initialBodyForwardWorld = forward;
    }

    private Vector3 GetInitialBoneAxisLocal(Transform bone, Transform childBone)
    {
        Vector3 worldDir = (childBone.position - bone.position).normalized;
        return bone.parent.InverseTransformDirection(worldDir).normalized;
    }

    [ContextMenu("播放")]
    public void Play()
    {
        currentTime = 0f;
        currentFrameIndex = 0;
        isPlaying = true;
        ResetBones();
    }

    [ContextMenu("暂停")]
    public void Pause()
    {
        isPlaying = false;
    }

    [ContextMenu("停止并重置")]
    public void StopAndReset()
    {
        isPlaying = false;
        currentTime = 0f;
        currentFrameIndex = 0;
        ResetBones();
    }

    [ContextMenu("重置骨骼")]
    public void ResetBones()
    {
        if (leftUpperArmBone != null) leftUpperArmBone.localRotation = leftUpperArmInitialLocalRotation;
        if (leftLowerArmBone != null) leftLowerArmBone.localRotation = leftLowerArmInitialLocalRotation;
        if (rightUpperArmBone != null) rightUpperArmBone.localRotation = rightUpperArmInitialLocalRotation;
        if (rightLowerArmBone != null) rightLowerArmBone.localRotation = rightLowerArmInitialLocalRotation;
    }

    private void ApplyFrame(DriveFrameData frame)
    {
        Vector3 leftShoulderTarget = BodyLocalPointToWorld(frame.leftShoulder);
        Vector3 leftElbowHint = BodyLocalPointToWorld(frame.leftElbow);
        Vector3 leftWristTarget = BodyLocalPointToWorld(frame.leftWrist);

        Vector3 rightShoulderTarget = BodyLocalPointToWorld(frame.rightShoulder);
        Vector3 rightElbowHint = BodyLocalPointToWorld(frame.rightElbow);
        Vector3 rightWristTarget = BodyLocalPointToWorld(frame.rightWrist);

        ConstrainArmTargets(true, leftShoulderTarget, ref leftElbowHint, ref leftWristTarget);
        ConstrainArmTargets(false, rightShoulderTarget, ref rightElbowHint, ref rightWristTarget);

        SolveArmTwoBoneIK(leftUpperArmBone, leftLowerArmBone, leftHandBone,
            leftUpperArmInitialLocalRotation, leftLowerArmInitialLocalRotation,
            leftUpperArmInitialAxisLocal, leftLowerArmInitialAxisLocal,
            leftUpperLength, leftLowerLength, leftElbowHint, leftWristTarget);

        SolveArmTwoBoneIK(rightUpperArmBone, rightLowerArmBone, rightHandBone,
            rightUpperArmInitialLocalRotation, rightLowerArmInitialLocalRotation,
            rightUpperArmInitialAxisLocal, rightLowerArmInitialAxisLocal,
            rightUpperLength, rightLowerLength, rightElbowHint, rightWristTarget);
    }

    private Vector3 BodyLocalPointToWorld(Vec3Data p)
    {
        float sx = invertRight ? -1f : 1f;
        float sy = invertUp ? -1f : 1f;
        float sz = invertForward ? -1f : 1f;

        Vector3 local = new Vector3(p.x * sx, p.y * sy, p.z * sz);
        float scale = modelShoulderWidth * positionScaleMultiplier;

        return initialShoulderMidWorld +
               initialBodyRightWorld * (local.x * scale) +
               initialBodyUpWorld * (local.y * scale) +
               initialBodyForwardWorld * (local.z * scale);
    }

    private void ConstrainArmTargets(bool isLeft, Vector3 shoulderWorld, ref Vector3 elbowHintWorld, ref Vector3 wristTargetWorld)
    {
        Vector3 shoulderMid = initialShoulderMidWorld;
        Vector3 hipMid = GetInitialHipMidWorld();
        Vector3 bodyDown = (hipMid - shoulderMid).normalized;
        Vector3 extendedHip = hipMid + bodyDown * (modelShoulderWidth * 0.25f);

        float bodyRadius = modelShoulderWidth * bodyRadiusScale;
        float wristMinDist = bodyRadius + modelShoulderWidth * wristClearanceScale;
        float elbowMinDist = bodyRadius * 0.95f;

        Vector3 sideDir = isLeft ? -initialBodyRightWorld : initialBodyRightWorld;
        Vector3 forwardDir = initialBodyForwardWorld;

        wristTargetWorld += forwardDir * (modelShoulderWidth * wristForwardBiasScale);

        Vector3 wristClosest = ClosestPointOnSegment(shoulderMid, extendedHip, wristTargetWorld);
        Vector3 wristDiff = wristTargetWorld - wristClosest;
        float wristDist = wristDiff.magnitude;
        if (wristDist < wristMinDist)
        {
            Vector3 pushDir = wristDist > 1e-6f ? wristDiff.normalized : (sideDir * 0.7f + forwardDir * 0.3f).normalized;
            wristTargetWorld = wristClosest + pushDir * wristMinDist;
        }

        elbowHintWorld += sideDir * (modelShoulderWidth * elbowOutwardBiasScale);
        Vector3 elbowClosest = ClosestPointOnSegment(shoulderMid, extendedHip, elbowHintWorld);
        Vector3 elbowDiff = elbowHintWorld - elbowClosest;
        float elbowDist = elbowDiff.magnitude;
        if (elbowDist < elbowMinDist)
        {
            Vector3 pushDir = elbowDist > 1e-6f ? elbowDiff.normalized : sideDir.normalized;
            elbowHintWorld = elbowClosest + pushDir * elbowMinDist;
        }
    }

    private void SolveArmTwoBoneIK(
        Transform upperBone,
        Transform lowerBone,
        Transform handBone,
        Quaternion upperInitialLocalRotation,
        Quaternion lowerInitialLocalRotation,
        Vector3 upperInitialAxisLocal,
        Vector3 lowerInitialAxisLocal,
        float upperLen,
        float lowerLen,
        Vector3 elbowHintWorld,
        Vector3 wristTargetWorld)
    {
        if (upperBone == null || lowerBone == null || handBone == null) return;

        Vector3 shoulderPos = upperBone.position;
        Vector3 shoulderToTarget = wristTargetWorld - shoulderPos;
        float rawDistance = shoulderToTarget.magnitude;
        if (rawDistance < 1e-6f) return;

        float minReach = Mathf.Abs(upperLen - lowerLen) + 0.0001f;
        float maxReach = upperLen + lowerLen - 0.0001f;
        float targetDistance = Mathf.Clamp(rawDistance, minReach, maxReach);

        Vector3 targetDir = shoulderToTarget.normalized;
        Vector3 clampedTarget = shoulderPos + targetDir * targetDistance;

        Vector3 hintDir = Vector3.ProjectOnPlane(elbowHintWorld - shoulderPos, targetDir);
        if (hintDir.sqrMagnitude < 1e-8f)
        {
            hintDir = Vector3.ProjectOnPlane(lowerBone.position - shoulderPos, targetDir);
        }
        if (hintDir.sqrMagnitude < 1e-8f) return;
        hintDir.Normalize();

        float cosAngle = Mathf.Clamp(
            (upperLen * upperLen + targetDistance * targetDistance - lowerLen * lowerLen) /
            (2f * upperLen * targetDistance), -1f, 1f);
        float sinAngle = Mathf.Sqrt(Mathf.Max(0f, 1f - cosAngle * cosAngle));

        Vector3 elbowPos = shoulderPos + targetDir * (upperLen * cosAngle) + hintDir * (upperLen * sinAngle);

        Vector3 upperWorldDir = (elbowPos - shoulderPos).normalized;
        ApplyBoneDirection(upperBone, upperInitialLocalRotation, upperInitialAxisLocal, upperWorldDir, rotationLerpSpeed);

        Vector3 newLowerPos = lowerBone.position;
        Vector3 lowerWorldDir = (clampedTarget - newLowerPos).normalized;
        ApplyBoneDirection(lowerBone, lowerInitialLocalRotation, lowerInitialAxisLocal, lowerWorldDir, rotationLerpSpeed);
    }

    private void ApplyBoneDirection(Transform bone, Quaternion initialLocalRotation, Vector3 initialAxisLocal, Vector3 targetWorldDirection, float lerpSpeed)
    {
        if (bone == null || bone.parent == null) return;
        if (targetWorldDirection.sqrMagnitude < 1e-8f) return;

        Vector3 targetDirParentLocal = bone.parent.InverseTransformDirection(targetWorldDirection).normalized;
        if (targetDirParentLocal.sqrMagnitude < 1e-8f) return;

        Quaternion delta = Quaternion.FromToRotation(initialAxisLocal, targetDirParentLocal);
        Quaternion targetLocalRotation = delta * initialLocalRotation;
        bone.localRotation = Quaternion.Slerp(bone.localRotation, targetLocalRotation, Time.deltaTime * lerpSpeed);
    }

    private Vector3 GetInitialHipMidWorld()
    {
        if (leftHipBone != null && rightHipBone != null)
        {
            return (leftHipBone.position + rightHipBone.position) * 0.5f;
        }
        return initialShoulderMidWorld - initialBodyUpWorld * (modelShoulderWidth * 1.2f);
    }

    private Vector3 ClosestPointOnSegment(Vector3 a, Vector3 b, Vector3 p)
    {
        Vector3 ab = b - a;
        float abLenSq = ab.sqrMagnitude;
        if (abLenSq < 1e-8f) return a;
        float t = Vector3.Dot(p - a, ab) / abLenSq;
        t = Mathf.Clamp01(t);
        return a + ab * t;
    }

    [Serializable]
    public class Vec3Data
    {
        public float x;
        public float y;
        public float z;
    }

    [Serializable]
    public class DriveFrameData
    {
        public Vec3Data leftShoulder;
        public Vec3Data leftElbow;
        public Vec3Data leftWrist;
        public Vec3Data rightShoulder;
        public Vec3Data rightElbow;
        public Vec3Data rightWrist;
    }

    [Serializable]
    public class DriveClipData
    {
        public float fps;
        public int frameCount;
        public string coordinateSystem;
        public string origin;
        public DriveFrameData[] frames;
    }
}
