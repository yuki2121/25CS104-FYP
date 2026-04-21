import { useEffect } from 'react'
import * as THREE from 'three';
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Button from './components/Button';
import './PoseEditorCanvas.css';



export let camera: THREE.PerspectiveCamera | null = null;
export let skeleton: THREE.Skeleton | null = null;

function PoseEditorCanvas() {

    const IK_CHAINS: Record<string, string[]> = {
        mixamorigRightHand: ["mixamorigRightArm", "mixamorigRightForeArm", "mixamorigRightHand"],
        mixamorigRightForeArm: ["mixamorigRightArm", "mixamorigRightForeArm"],
        mixamorigLeftHand:  ["mixamorigLeftArm",  "mixamorigLeftForeArm",  "mixamorigLeftHand"],
        mixamorigLeftForeArm:  ["mixamorigLeftArm",  "mixamorigLeftForeArm"],
        mixamorigRightFoot: ["mixamorigRightUpLeg","mixamorigRightLeg",    "mixamorigRightFoot"],
        mixamorigRightLeg:  ["mixamorigRightUpLeg","mixamorigRightLeg"],
        mixamorigLeftFoot:  ["mixamorigLeftUpLeg", "mixamorigLeftLeg",     "mixamorigLeftFoot"],
        mixamorigLeftLeg:   ["mixamorigLeftUpLeg", "mixamorigLeftLeg"],
        mixamorigHead:      ["mixamorigNeck",      "mixamorigHead"],  
        mixamorigNeck:      [ "mixamorigSpine2",    "mixamorigNeck"],
        mixamorigSpine2:    ["mixamorigSpine1",  "mixamorigSpine2"],
        mixamorigSpine1:    ["mixamorigSpine",   "mixamorigSpine1"],
        
        mixamorigSpine:     ["mixamorigHips",     "mixamorigSpine"],
        // mixamorigHips:     ["mixamorigSpine",     "mixamorigHips"],
        mixamorigRightArm: ["mixamorigRightShoulder","mixamorigRightArm"],
        mixamorigLeftArm:  ["mixamorigLeftShoulder", "mixamorigLeftArm"],
        mixamorigRightToe_End: ["mixamorigRightFoot",  "mixamorigRightToe_End"],
        mixamorigLeftToe_End: ["mixamorigLeftFoot", "mixamorigLeftToe_End"],
        // mixamorigRightUpLeg: ["mixamorigSpine","mixamorigHips", "mixamorigRightUpLeg"],
        // mixamorigLeftUpLeg:  ["mixamorigSpine","mixamorigHips", "mixamorigLeftUpLeg"],
        // mixamorigRightMiddle1: ["mixamorigRightHand", "mixamorigRightHandIndex1","mixamorigRightHandMiddle1"],
        // mixamorigLeftMiddle1:  ["mixamorigLeftHand",  "mixamorigLeftHandMiddle1"],

    };

    const defaultBoneRotations: Record<string, THREE.Quaternion> = {};


    const sprites: THREE.Sprite[] = [];

    let cameraControls: OrbitControls | null = null;


    let skeletonHelper: THREE.SkeletonHelper | null = null;

    let resultDisplay = null;

    // IK Solver helper
    function ccdSolve(
        links: THREE.Bone[],
        effector: THREE.Bone,
        targetWorld: THREE.Vector3,
        iterations = 12,
        eps = 1e-3
    ) {
        const linkPos = new THREE.Vector3();
        const effPos = new THREE.Vector3();
        const tgtPos = new THREE.Vector3();

        const invLinkQ = new THREE.Quaternion();
        const deltaQ = new THREE.Quaternion();

        const toEff = new THREE.Vector3();
        const toTgt = new THREE.Vector3();
        const axis = new THREE.Vector3();

        for (let it = 0; it < iterations; it++) {
            effector.getWorldPosition(effPos);
            if (effPos.distanceToSquared(targetWorld) < eps * eps) break;

            for (let i = links.length - 1; i >= 0; i--) {
                const link = links[i];

                link.getWorldPosition(linkPos);
                link.getWorldQuaternion(invLinkQ).invert();

                effector.getWorldPosition(effPos);
                tgtPos.copy(targetWorld);

                // move vectors into link local rotation space
                toEff.copy(effPos).sub(linkPos).applyQuaternion(invLinkQ);
                toTgt.copy(tgtPos).sub(linkPos).applyQuaternion(invLinkQ);

                const len1 = toEff.length();
                const len2 = toTgt.length();
                if (len1 < 1e-8 || len2 < 1e-8) continue;

                toEff.multiplyScalar(1 / len1);
                toTgt.multiplyScalar(1 / len2);

                let cos = THREE.MathUtils.clamp(toEff.dot(toTgt), -1, 1);
                const angle = Math.acos(cos);
                if (angle < 1e-5) continue;

                axis.crossVectors(toEff, toTgt);

                // fallback when nearly parallel/opposite
                if (axis.lengthSq() < 1e-10) {
                    axis.set(1, 0, 0).cross(toEff);
                    if (axis.lengthSq() < 1e-10) axis.set(0, 1, 0).cross(toEff);
                }

                axis.normalize();
                deltaQ.setFromAxisAngle(axis, angle);

                // apply in local space
                link.quaternion.multiply(deltaQ);
                link.quaternion.normalize();

                link.updateMatrixWorld(true);
            }
        }
    }

    function resetBonePositions() {
        if (!skeleton) return;
        if (Object.keys(defaultBoneRotations).length === 0) return;
        for (const boneName in defaultBoneRotations) {
            const bone = skeleton.getBoneByName(boneName);
            if (bone) {
                const defRot = defaultBoneRotations[boneName];
                bone.quaternion.copy(defRot);
            }
        }
        updateAllSpritePositions();
    }

    function updateAllSpritePositions() {
        if (!skeleton) return;
        if (sprites.length === 0) return;
        const tmp = new THREE.Vector3();
        for (const s of sprites) {
            const boneName = s.name.replace("_Control", "");
            const bone = skeleton.getBoneByName(boneName);
            if (!bone) continue;

            bone.getWorldPosition(tmp);
            s.position.copy(tmp);
        }
    }

    function resetCameraPosition() {
        if (!cameraControls) return;
        cameraControls.reset();
        cameraControls.target.set(0, 90, 0);
    }

    function toggleSpriteVisibility() {
        for (const s of sprites) {
            const mat = s.material as THREE.SpriteMaterial;
            if (mat.opacity > 0) {
                mat.opacity = 0;
            } else {
                mat.opacity = 0.6;
            }
        }
    }

    function toggleSkeletonVisibility() {
        if (skeletonHelper) {
            skeletonHelper.visible = !skeletonHelper.visible;
        }
    }




    useEffect(() => {
        const scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(40, 1, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('pose-editor-canvas') as HTMLCanvasElement });
        renderer.setSize(500, 500);
        renderer.setPixelRatio(window.devicePixelRatio);
        camera.position.z = 300;
        camera.position.x = 0;
        camera.position.y = 90;
        // camera.lookAt(250, 250, 0);

        cameraControls = new OrbitControls(camera, renderer.domElement);
        cameraControls.target.set(0, 90, 0);
        cameraControls.enableDamping = true;
        cameraControls.dampingFactor = 0.05;
        cameraControls.enablePan = false;
        cameraControls.mouseButtons["LEFT"] = null;
        cameraControls.mouseButtons["RIGHT"] = THREE.MOUSE.ROTATE;
        cameraControls.update();



        //set light and background
        renderer.setClearColor(0xdddddd, 1);
        const Dlight = new THREE.DirectionalLight(0xffffff, 1);
        Dlight.position.set(0, 1, 1).normalize();
        
        const Alight = new THREE.AmbientLight(0x404040, 0.3); 
        scene.add(Dlight);
        scene.add(Alight);

        // load model and set bone to be drag
        const loader = new GLTFLoader();
        loader.load('/models/model.glb', onLoad, onProgress, onError)

        const bones = ["mixamorigRightArm", 
            "mixamorigRightForeArm", 
            "mixamorigRightHand",
            // "mixamorigRightShoulder",
            // "mixamorigRightHandMiddle1" ,
                "mixamorigLeftArm", 
                "mixamorigLeftForeArm", 
                "mixamorigLeftHand",
                // "mixamorigLeftHandMiddle1" ,
                // "mixamorigLeftShoulder",
                // "mixamorigRightUpLeg", 
                "mixamorigRightLeg", 
                "mixamorigRightFoot", 
                "mixamorigRightToe_End",
                // "mixamorigLeftUpLeg", 
                "mixamorigLeftLeg", 
                "mixamorigLeftFoot", 
                "mixamorigLeftToe_End",
                "mixamorigHead", 
                "mixamorigNeck",  
                "mixamorigSpine1",
                "mixamorigSpine2",
                "mixamorigSpine",
                // "mixamorigHips"
            
            ];



        let selectedSprite: THREE.Sprite | null = null;







        //animation loop
        let dragging = false;
        let rafId = 0;
        const animate = () => {
            rafId = requestAnimationFrame(animate);
            scene.updateMatrixWorld(true);
            
            if (dragging && skeleton) {
                updateAllSpritePositions();
            }

            if (!camera || !cameraControls) return;

            cameraControls.update();
            Dlight.position.copy(camera.position).normalize();
            renderer.render(scene, camera);

        };
        animate();




        function onLoad(gltf: any) {



            //load mesh
            const model = gltf.scene;
            model.scale.set(100, 100, 100);
            model.position.set(0, 0, 0);
            scene.add(model);
            skeletonHelper = new THREE.SkeletonHelper( model );
            skeletonHelper.visible = true;

            //find skinned mesh
            const skinned: THREE.SkinnedMesh[] = [];
            
            gltf.scene.traverse((obj: any)=> {
                if ((obj as any).isSkinnedMesh) skinned.push(obj as THREE.SkinnedMesh);
            });
            console.log("skinned meshes:", skinned);

            const body = skinned[0];
            skeleton = body.skeleton;
            console.log("Bone names:");
            skeleton.bones.forEach((b, i) => console.log(i, b.name));

            scene.add(skeletonHelper);

            gltf.scene.updateMatrixWorld(true);

            //set default bone positions
            skeleton.bones.forEach( (b) => {
                const bone = skeleton!.getBoneByName(b.name);
                if (bone) {
                    const rot = new THREE.Quaternion();
                    rot.copy(bone.quaternion);
                    defaultBoneRotations[bone.name] = rot;
                }
            });


            //drag controls
            const textureLoader = new THREE.TextureLoader();
            const dragIcon = textureLoader.load( '/icons8-circle-48-2.png' );

            bones.forEach( (boneName) => {
                if (!skeleton) return;
                const bone = skeleton.getBoneByName(boneName);
                const sprite = new THREE.Sprite(new THREE.SpriteMaterial( { map: dragIcon, transparent: true, depthTest: false, opacity: 0.6 } ));

                sprite.scale.set(8, 8, 1);
                sprite.position.set(bone!.getWorldPosition(new THREE.Vector3()).x,
                                    bone!.getWorldPosition(new THREE.Vector3()).y,
                                    bone!.getWorldPosition(new THREE.Vector3()).z);
                
                sprite.name = boneName + "_Control";
                
                sprites.push(sprite);
                scene.add(sprite);
            });

            // drag control for sprite
            renderer.domElement.style.touchAction = "none";

            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            const dragPlane = new THREE.Plane();
            const hitPoint = new THREE.Vector3();
            const dragOffset = new THREE.Vector3();
            const camDir = new THREE.Vector3();

            function updateMouseNDC(e: PointerEvent) {
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
            }



            function onPointerDown(e: PointerEvent) {
                updateMouseNDC(e);
                if (!camera) return;
                raycaster.setFromCamera(mouse, camera);

                const hits = raycaster.intersectObjects(sprites, false);
                if (hits.length === 0) return;

                selectedSprite = hits[0].object as THREE.Sprite;
                dragging = true;

                // set drag plane para to camera
                camera.getWorldDirection(camDir);
                dragPlane.setFromNormalAndCoplanarPoint(camDir, hits[0].point);
                dragOffset.copy(hits[0].point).sub(selectedSprite.position);
                renderer.domElement.setPointerCapture(e.pointerId);

                // tint selected sprite to blueee
                const mat = selectedSprite.material as THREE.SpriteMaterial;
                mat.color.set(0x00aaff);
            }

            function onPointerMove(e: PointerEvent) {
                if (!dragging || !selectedSprite || !camera) return;

                updateMouseNDC(e);
                raycaster.setFromCamera(mouse, camera);

                if (raycaster.ray.intersectPlane(dragPlane, hitPoint)) {
                    //update sprite
                    selectedSprite.position.copy(hitPoint).sub(dragOffset);
                    //update bone with IK
                    const ik = selectedSprite ? getIKForSprite(selectedSprite) : null;
                    if (ik) {
                        ccdSolve(ik.links, ik.effector, selectedSprite.position.clone(), 5);
                        body.updateMatrixWorld(true);
                    }

                }
            }

            function onPointerUp(e: PointerEvent) {
                if (!dragging) return;
                dragging = false;

                // reset sprite color
                if (selectedSprite) {
                    const mat = selectedSprite.material as THREE.SpriteMaterial;
                    mat.color.set(0xffffff); 
                }
                selectedSprite = null;

                try {
                    renderer.domElement.releasePointerCapture(e.pointerId);
                } catch {}
            }

            const el = renderer.domElement;
            el.addEventListener("pointerdown", onPointerDown);
            el.addEventListener("pointermove", onPointerMove);
            el.addEventListener("pointerup", onPointerUp);
            el.addEventListener("pointerleave", onPointerUp);
            el.addEventListener("pointercancel", onPointerUp);


            // map sprite to IK
            function getIKForSprite(sprite: THREE.Sprite) {
                if (!skeleton) return null;
                const effName = sprite.name.replace("_Control", "");

                const chainNames = IK_CHAINS[effName];
                if (!chainNames) return null;

                const chainBones = chainNames
                    .map(n => skeleton!.getBoneByName(n))
                    .filter(Boolean) as THREE.Bone[];

                if (chainBones.length < 2) return null;

                const effector = chainBones[chainBones.length - 1];
                const links = chainBones.slice(0, chainBones.length - 1); 

                return { effName, effector, links };
            }



        }

        function onProgress(xhr: any) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        }

        function onError(error: any) {
            console.error('An error happened', error);
        }


        return () => {
            cancelAnimationFrame(rafId);
            renderer.dispose();
        };


    }, []);

    const handleSearchClick = () => {
        window.dispatchEvent(new CustomEvent('trigger-pose-search'));
    };

    return (
        <div>
            <div>
                <canvas id="pose-editor-canvas" width={300} height={500} style={{ border: '2px solid black' }} />
                
            </div>
            <div>
                <Button label="Reset Pose" onClick={resetBonePositions} />
                <Button label="Reset Camera" onClick={resetCameraPosition}/>

            </div>
            <div>
                <Button label="Toggle Sprite" onClick={toggleSpriteVisibility} />
                <Button label="Toggle Skeleton" onClick={toggleSkeletonVisibility} />
            </div>

            <div style={{ marginTop: '5px', borderTop: '1px solid #ccc', paddingTop: '5px' }}>
                <Button 
                    label="Search Poses" 
                    onClick={handleSearchClick} 
                    className="bg-green-500" 
                    // style={{ width: '100%', height: '50px', fontWeight: 'bold', fontSize: '1.1rem' }} 
                />
            </div>
            <div style = {{ fontSize: '0.6rem', color: 'gray', marginTop: '5px' }}>
                Notice: This is a non-commercial academic project. The searchable index contains images used for demonstration and evaluation purposes only under Fair Use. These images are utilized for indexing and retrieval and are not used for generative AI training. If you are a copyright holder and wish to have an image removed, please contact puikuklau2-c@my.cityu.edu.hk.
            </div>

            {resultDisplay}
        </div>
    );
}

export default PoseEditorCanvas;