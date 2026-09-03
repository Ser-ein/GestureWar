using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public enum ActiveSlot
{
    Primary,
    Secondary,
    Empty
}

public class GunManager : MonoBehaviour
{
    public Transform playerCamera;
    public Vector3 holdPosition = new Vector3(0.5f, -0.2f, 0.8f);
    public Vector3 holdRotation = new Vector3(0f, 0f, 0f);

    [Header("交互设置")]
    public KeyCode interactKey = KeyCode.E;
    public float maxPickUpDistance = 2f;
    public int hitboxLayer = 10;
    public int blockLayer = 9;
    public KeyCode fireKey = KeyCode.Mouse0;
    public KeyCode reloadKey = KeyCode.R;
    public KeyCode dropKey = KeyCode.F;
    public float dropForwardSpeed = 4f;

    [Header("栏位/切换")]
    public KeyCode slot1Key = KeyCode.Alpha1;
    public KeyCode slot2Key = KeyCode.Alpha2;
    public KeyCode slot3Key = KeyCode.Alpha3;
    public string emptyHandAnimParameter = "Status_pistol";

    private Animator animator;
    private bool hasWeaponAnimParameter;
    private string activeAnimParameter;

    private GameObject primaryGun = null;
    private GameObject secondaryGun = null;
    private ActiveSlot activeSlot = ActiveSlot.Empty;

    private readonly Dictionary<GameObject, CharacterModels> weaponModels = new Dictionary<GameObject, CharacterModels>();

    struct CharacterModels
    {
        public Transform characterModel;
        public Transform shadowModel;
    }

    public GameObject CurrentActiveGun => GetGunInSlot(activeSlot);
    public GameObject GetCurrentGun() => CurrentActiveGun;

    private PickUpData lastTargetedPickUpData = null;

    void Start()
    {
        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        UpdateActiveAnimParameter();
        ApplyActiveSlot();
    }

    void Update()
    {
        if (Input.GetKeyDown(slot1Key) && GetGunInSlot(ActiveSlot.Primary) != null) SwitchToSlot(ActiveSlot.Primary);
        if (Input.GetKeyDown(slot2Key) && GetGunInSlot(ActiveSlot.Secondary) != null) SwitchToSlot(ActiveSlot.Secondary);
        if (Input.GetKeyDown(slot3Key)) SwitchToSlot(ActiveSlot.Empty);

        if (Input.GetKeyDown(dropKey) && CurrentActiveGun != null)
        {
            DropGun();
        }

        HandlePickUpDetection();
    }

    // =================== 新增公开换武器方法 ===================
    /// <summary>
    /// 按顺序切换到下一个可用的武器栏位（主→副→空手→主……），
    /// 若某个栏位为空则跳过，最终回到原栏位。
    /// 此方法供外部（如 UDPReceiver）通过 UnityEvent 调用。
    /// </summary>
    public void SwitchWeapon()
    {
        // 获取当前激活的槽位索引（0=主,1=副,2=空手）
        int currentIndex = (int)activeSlot;
        // 槽位数组顺序：主、副、空手
        ActiveSlot[] slotOrder = { ActiveSlot.Primary, ActiveSlot.Secondary, ActiveSlot.Empty };
        int count = slotOrder.Length;

        // 从下一个槽位开始查找
        for (int i = 1; i <= count; i++)
        {
            int nextIndex = (currentIndex + i) % count;
            ActiveSlot nextSlot = slotOrder[nextIndex];
            // 如果该槽位有武器，或者为空手（空手总是可用的），则切换过去
            if (nextSlot == ActiveSlot.Empty || GetGunInSlot(nextSlot) != null)
            {
                SwitchToSlot(nextSlot);
                return;
            }
        }
        // 理论上不可能所有槽位都不可用（空手总可用），但以防万一：
        // 如果当前已经是空手且没有其他武器，则保持在空手
        if (activeSlot != ActiveSlot.Empty)
            SwitchToSlot(ActiveSlot.Empty);
    }
    // ======================================================

    void HandlePickUpDetection()
    {
        if (Camera.main == null) return;

        Ray ray = Camera.main.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
        RaycastHit hit;
        PickUpData targeted = null;

        int mask = ~0 & ~(1 << Mathf.Clamp(hitboxLayer, 0, 31));
        if (Physics.Raycast(ray, out hit, maxPickUpDistance, mask, QueryTriggerInteraction.Collide))
        {
            PickUpData data = hit.transform.GetComponentInParent<PickUpData>();
            if (data != null && data.gameObject != CurrentActiveGun)
                targeted = data;
        }

        if (targeted != lastTargetedPickUpData)
        {
            if (lastTargetedPickUpData != null)
                lastTargetedPickUpData.HidePrompt();
            lastTargetedPickUpData = targeted;
            if (targeted != null)
                targeted.ShowPrompt();
        }

        if (targeted != null && Input.GetKeyDown(interactKey))
        {
            PickUpGun(targeted.gameObject);
        }
    }

    public void PickUpGun(GameObject gunToPickUp)
    {
        if (gunToPickUp == null) return;
        GunData gunData = gunToPickUp.GetComponent<GunData>();
        ActiveSlot targetSlot = (gunData != null && gunData.weaponSlot == WeaponSlot.Secondary)
            ? ActiveSlot.Secondary : ActiveSlot.Primary;

        GameObject existing = GetGunInSlot(targetSlot);
        if (existing != null)
        {
            HideCharacterModelsFor(existing);
            if (existing.TryGetComponent(out GunData existingData))
                ResetAnimParameterForced(existingData.weaponAnimParameter);
            ApplyWorldState(existing);
            weaponModels.Remove(existing);
        }

        EquipWeapon(gunToPickUp);
        CaptureCharacterModels(gunToPickUp);
        SetSlot(targetSlot, gunToPickUp);

        ApplyActiveSlot();

        if (gunToPickUp.TryGetComponent(out PickUpData data))
            data.HidePrompt();
    }

    void DropGun()
    {
        GameObject gun = CurrentActiveGun;
        if (gun == null) return;

        HideCharacterModelsFor(gun);
        ApplyWorldState(gun);
        weaponModels.Remove(gun);
        SetSlot(activeSlot, null);

        if (gun.TryGetComponent(out GunData droppedData))
            ResetAnimParameterForced(droppedData.weaponAnimParameter);

        SwitchToSlot(GetFirstOccupiedSlot());

        Rigidbody rb = gun.GetComponent<Rigidbody>();
        if (rb != null)
        {
            Vector3 dropVelocity = playerCamera.forward * dropForwardSpeed;
            CharacterMoveAndLook controller = GetComponent<CharacterMoveAndLook>();
            if (controller != null)
            {
                Vector3 playerVelocity = controller.HorizontalVelocity;
                if (playerVelocity.magnitude > 0.01f)
                    dropVelocity += playerVelocity;
            }
            rb.velocity = dropVelocity;
        }
    }

    GameObject GetGunInSlot(ActiveSlot slot)
    {
        return slot == ActiveSlot.Primary ? primaryGun : slot == ActiveSlot.Secondary ? secondaryGun : null;
    }

    void SetSlot(ActiveSlot slot, GameObject gun)
    {
        if (slot == ActiveSlot.Primary) primaryGun = gun;
        else if (slot == ActiveSlot.Secondary) secondaryGun = gun;
    }

    ActiveSlot GetFirstOccupiedSlot()
    {
        if (primaryGun != null) return ActiveSlot.Primary;
        if (secondaryGun != null) return ActiveSlot.Secondary;
        return ActiveSlot.Empty;
    }

    void SwitchToSlot(ActiveSlot slot)
    {
        activeSlot = slot;
        ApplyActiveSlot();
    }

    void ApplyActiveSlot()
    {
        UpdateActiveAnimParameter();

        GameObject activeGun = GetGunInSlot(activeSlot);
        SetActiveSafe(primaryGun, activeSlot == ActiveSlot.Primary || (primaryGun == activeGun));
        SetActiveSafe(secondaryGun, activeSlot == ActiveSlot.Secondary || (secondaryGun == activeGun));

        SyncWeaponModelsVisibility(activeGun);
        SetWeaponAnimState(2);
    }

    void SyncWeaponModelsVisibility(GameObject activeGun)
    {
        foreach (var kv in weaponModels)
        {
            if (kv.Key == null) continue;
            bool visible = kv.Key == activeGun;
            SetActiveSafe(kv.Value.characterModel?.gameObject, visible);
            SetActiveSafe(kv.Value.shadowModel?.gameObject, visible);
        }
    }

    static void SetActiveSafe(GameObject go, bool active)
    {
        if (go != null && go.activeSelf != active)
            go.SetActive(active);
    }

    void HideCharacterModelsFor(GameObject gun)
    {
        if (gun != null && weaponModels.TryGetValue(gun, out CharacterModels models))
        {
            SetActiveSafe(models.characterModel?.gameObject, false);
            SetActiveSafe(models.shadowModel?.gameObject, false);
        }
    }

    void UpdateActiveAnimParameter()
    {
        if (activeSlot == ActiveSlot.Empty)
        {
            activeAnimParameter = emptyHandAnimParameter;
        }
        else
        {
            GunData data = GetGunInSlot(activeSlot)?.GetComponent<GunData>();
            activeAnimParameter = data != null ? data.weaponAnimParameter : emptyHandAnimParameter;
        }
    }

    public void SetWeaponAnimState(int state)
    {
        if (animator == null) return;

        ResetOtherWeaponAnimParameters();

        hasWeaponAnimParameter = false;
        if (!string.IsNullOrEmpty(activeAnimParameter) && animator.parameters != null)
        {
            foreach (AnimatorControllerParameter p in animator.parameters)
            {
                if (p.name == activeAnimParameter)
                {
                    hasWeaponAnimParameter = true;
                    break;
                }
            }
        }

        if (hasWeaponAnimParameter)
            animator.SetInteger(activeAnimParameter, state);
    }

    void ResetOtherWeaponAnimParameters()
    {
        GameObject activeGun = GetGunInSlot(activeSlot);

        if (primaryGun != null && primaryGun != activeGun && primaryGun.TryGetComponent(out GunData primaryData))
        {
            ResetAnimParameter(primaryData.weaponAnimParameter);
        }
        if (secondaryGun != null && secondaryGun != activeGun && secondaryGun.TryGetComponent(out GunData secondaryData))
        {
            ResetAnimParameter(secondaryData.weaponAnimParameter);
        }
        if (activeSlot != ActiveSlot.Empty)
        {
            ResetAnimParameter(emptyHandAnimParameter);
        }
    }

    void ResetAnimParameter(string paramName)
    {
        if (string.IsNullOrEmpty(paramName)) return;
        if (paramName == activeAnimParameter) return;
        ResetAnimParameterForced(paramName);
    }

    void ResetAnimParameterForced(string paramName)
    {
        if (string.IsNullOrEmpty(paramName)) return;
        if (animator == null || animator.parameters == null) return;

        foreach (AnimatorControllerParameter p in animator.parameters)
        {
            if (p.name == paramName)
            {
                animator.SetInteger(paramName, 0);
                break;
            }
        }
    }

    void ApplyWorldState(GameObject gun)
    {
        if (gun == null) return;

        gun.transform.SetParent(null);
        gun.SetActive(true);

        Rigidbody rb = gun.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = false;
            rb.useGravity = true;
        }

        GunData data = gun.GetComponent<GunData>();
        foreach (Collider col in gun.GetComponents<Collider>())
        {
            bool isPickupTrigger = data != null && data.pickupTrigger == col;
            col.isTrigger = isPickupTrigger;
        }

        foreach (Renderer ren in gun.GetComponentsInChildren<Renderer>())
            ren.shadowCastingMode = ShadowCastingMode.On;

        if (data != null) data.CancelReload();
    }

    void EquipWeapon(GameObject gun)
    {
        Rigidbody rb = gun.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = true;
            rb.useGravity = false;
        }

        foreach (Collider col in gun.GetComponents<Collider>())
            col.isTrigger = true;

        foreach (Renderer ren in gun.GetComponentsInChildren<Renderer>())
            ren.shadowCastingMode = ShadowCastingMode.Off;

        GunData data = gun.GetComponent<GunData>();
        Vector3 finalPosition = holdPosition + (data != null ? data.holdPositionOffset : Vector3.zero);
        Vector3 finalRotation = holdRotation + (data != null ? data.holdRotationOffset : Vector3.zero);

        gun.transform.SetParent(playerCamera);
        gun.transform.localPosition = finalPosition;
        gun.transform.localEulerAngles = finalRotation;
    }

    public GameObject SpawnWeapon(GameObject prefab, Vector3 position, Quaternion rotation)
    {
        if (prefab == null) return null;
        GameObject weapon = Instantiate(prefab, position, rotation);
        ApplyWorldState(weapon);
        return weapon;
    }

    public GameObject SpawnWeaponEquipped(GameObject prefab, WeaponSlot slot)
    {
        if (prefab == null || playerCamera == null) return null;

        GameObject weapon = Instantiate(prefab, playerCamera.position, playerCamera.rotation);
        ActiveSlot targetSlot = slot == WeaponSlot.Primary ? ActiveSlot.Primary : ActiveSlot.Secondary;
        GameObject existing = GetGunInSlot(targetSlot);
        if (existing != null)
        {
            HideCharacterModelsFor(existing);
            ApplyWorldState(existing);
            weaponModels.Remove(existing);
        }

        EquipWeapon(weapon);
        CaptureCharacterModels(weapon);
        SetSlot(targetSlot, weapon);

        ApplyActiveSlot();

        return weapon;
    }

    void CaptureCharacterModels(GameObject gun)
    {
        GunData data = gun.GetComponent<GunData>();
        string keyword = data != null ? data.characterModelName : "";

        CharacterModels models = new CharacterModels();
        if (!string.IsNullOrEmpty(keyword))
            models.characterModel = FindWeaponModelInChildren(transform, keyword);

        if (models.characterModel == null)
        {
            Debug.LogWarning($"未在角色模型上找到武器子物体：{gun.name}。请在 GunData.characterModelName 填写正确名称。");
        }

        ShadowProxySync sync = GetComponent<ShadowProxySync>();
        if (sync != null && sync.shadowProxy != null && models.characterModel != null)
            models.shadowModel = FindWeaponModelInChildren(sync.shadowProxy, models.characterModel.name);

        weaponModels[gun] = models;
    }

    Transform FindWeaponModelInChildren(Transform root, string weaponName)
    {
        if (root == null) return null;
        foreach (Transform child in root)
        {
            if (child.name.Contains(weaponName))
                return child;

            Transform result = FindWeaponModelInChildren(child, weaponName);
            if (result != null)
                return result;
        }
        return null;
    }
}