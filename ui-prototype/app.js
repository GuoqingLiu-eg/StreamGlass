import * as THREE from "./assets/vendor/three.module.min.js";

const languages = {
  cmn: { code: "CMN", name: "Chinese", country: "China", flag: "cn", lat: 39.9042, lon: 116.4074 },
  eng: { code: "ENG", name: "English", country: "United Kingdom", flag: "gb", lat: 51.5072, lon: -0.1276 },
  fra: { code: "FRA", name: "French", country: "France", flag: "fr", lat: 48.8566, lon: 2.3522 },
  deu: { code: "DEU", name: "German", country: "Germany", flag: "de", lat: 52.52, lon: 13.405 },
  spa: { code: "SPA", name: "Spanish", country: "Spain", flag: "es", lat: 40.4168, lon: -3.7038 },
  ita: { code: "ITA", name: "Italian", country: "Italy", flag: "it", lat: 41.9028, lon: 12.4964 },
  jpn: { code: "JPN", name: "Japanese", country: "Japan", flag: "jp", lat: 35.6762, lon: 139.6503 },
  kor: { code: "KOR", name: "Korean", country: "South Korea", flag: "kr", lat: 37.5665, lon: 126.978 },
  rus: { code: "RUS", name: "Russian", country: "Russia", flag: "ru", lat: 55.7558, lon: 37.6173 },
  por: { code: "POR", name: "Portuguese", country: "Portugal", flag: "pt", lat: 38.7223, lon: -9.1393 },
  arb: { code: "ARB", name: "Arabic", country: "Saudi Arabia", flag: "sa", lat: 24.7136, lon: 46.6753 },
  vie: { code: "VIE", name: "Vietnamese", country: "Vietnam", flag: "vn", lat: 21.0278, lon: 105.8342 },
};

function setFlagImage(image, language) {
  image.src = `./assets/flags/${language.flag}.svg`;
}

const targetLanguage = document.querySelector("#targetLanguage");
const languageMap = document.querySelector("#languageMap");
const selectFlag = document.querySelector("#selectFlag");
const selectCode = document.querySelector("#selectCode");
const selectShell = document.querySelector(".select-shell");
const mapFlag = document.querySelector("#mapFlag");
const mapCode = document.querySelector("#mapCode");
const mapName = document.querySelector("#mapName");
const mapCountry = document.querySelector("#mapCountry");
const globeViewport = document.querySelector("#globeViewport");
const globeCanvas = document.querySelector("#globeCanvas");
const globeCountryLabel = document.querySelector("#globeCountryLabel");
const globeLabelFlag = document.querySelector("#globeLabelFlag");
const globeLabelCountry = document.querySelector("#globeLabelCountry");
const globeSingaporeLabel = document.querySelector("#globeSingaporeLabel");
const listenControl = document.querySelector("#listenControl");
const listenButtonLabel = document.querySelector("#listenButtonLabel");
const listeningCopy = document.querySelector("#listeningCopy");
const bluetoothLensStatus = document.querySelector("#bluetoothLensStatus");
const runtimeStatus = document.querySelector("#runtimeStatus");
const runtimeStatusText = document.querySelector("#runtimeStatusText");
const speakerMirrorControl = document.querySelector("#speakerMirrorControl");
const speakerMirror = document.querySelector("#speakerMirror");
const speakerOutput = document.querySelector("#speakerOutput");
const speakerMirrorStatus = document.querySelector("#speakerMirrorStatus");
const speakerVolume = document.querySelector("#speakerVolume");
const speakerVolumeValue = document.querySelector("#speakerVolumeValue");
const translationCaptions = document.querySelector("#translationCaptions");
const captionLanguage = document.querySelector("#captionLanguage");
const captionPhase = document.querySelector("#captionPhase");
const captionFeed = document.querySelector("#captionFeed");
const captionPrevious = document.querySelector("#captionPrevious");
const captionCurrent = document.querySelector("#captionCurrent");
const toast = document.querySelector("#toast");
const culturalHero = document.querySelector(".cultural-hero");
const languageOptions = [
  ...languageMap.querySelectorAll(".globe-target[data-code]"),
];

let isPaused = false;
let runtimeOnline = false;
let statusRequestActive = false;
let playbackRequestActive = false;
let playbackRequestPending = false;
let playbackDesiredRevision = 0;
let playbackQueuedToast = "";
let playbackOutputsSignature = "";
let speakerVolumeRequestTimer;
let speakerVolumeEditing = false;
let confirmedLanguage = "eng";
let toastTimer;
let selectionTimer;
let captionLanguageCode = "eng";
let captionRevision = -1;
let captionSuppressThrough = -1;
let lastProcessedUtterances = 0;
let lastTranslationRevision = -1;
let currentCaptionUtterance = -1;
let currentCaptionFinal = true;
let currentCaption = "";
let captionHistory = [];
let captionScrollFrame;
let runtimeEvents;
let runtimeFallbackTimer;
let focusGlobeCountry = () => {};

const captionLanguageTags = {
  cmn: "zh",
  eng: "en",
  fra: "fr",
  deu: "de",
  spa: "es",
  ita: "it",
  jpn: "ja",
  kor: "ko",
  rus: "ru",
  por: "pt",
  arb: "ar",
  vie: "vi",
};

function clearTranslationCaptions(code, suppressThrough = -1) {
  window.cancelAnimationFrame(captionScrollFrame);
  captionLanguageCode = code;
  captionRevision = suppressThrough;
  captionSuppressThrough = suppressThrough;
  currentCaptionUtterance = -1;
  currentCaptionFinal = true;
  currentCaption = "";
  captionHistory = [];
  captionPrevious.replaceChildren();
  captionFeed.style.removeProperty("--caption-center-space");
  captionFeed.scrollTop = 0;
  captionCurrent.textContent = "Translation will appear here.";
  captionCurrent.classList.add("caption-placeholder");
  captionPhase.textContent = "Ready";
  translationCaptions.classList.remove("is-streaming", "is-final");
  translationCaptions.classList.add("is-idle");

  const language = languages[code];
  captionLanguage.textContent = language
    ? `${language.name} translation`
    : "Live translation";
  captionCurrent.lang = captionLanguageTags[code] || "";
  captionPrevious.lang = captionLanguageTags[code] || "";
}

function renderCaptionHistory() {
  const fragment = document.createDocumentFragment();
  captionHistory.forEach(({ text }) => {
    const line = document.createElement("p");
    line.dir = "auto";
    line.lang = captionLanguageTags[captionLanguageCode] || "";
    line.textContent = text;
    fragment.append(line);
  });
  captionPrevious.replaceChildren(fragment);
}

function centerCurrentCaption() {
  window.cancelAnimationFrame(captionScrollFrame);
  captionScrollFrame = window.requestAnimationFrame(() => {
    const currentLine = captionCurrent.closest(".caption-current");
    if (!currentLine) {
      return;
    }

    const centerSpace = Math.max(28, Math.round(captionFeed.clientHeight / 2));
    captionFeed.style.setProperty("--caption-center-space", `${centerSpace}px`);

    const feedBox = captionFeed.getBoundingClientRect();
    const lineBox = currentLine.getBoundingClientRect();
    const target =
      captionFeed.scrollTop +
      lineBox.top -
      feedBox.top -
      (captionFeed.clientHeight - lineBox.height) / 2;
    const maxScroll = Math.max(0, captionFeed.scrollHeight - captionFeed.clientHeight);
    captionFeed.scrollTop = Math.min(maxScroll, Math.max(0, target));
  });
}

function rememberCaption(text, utterance) {
  if (!text) {
    return;
  }
  const previous = captionHistory.at(-1);
  if (previous?.utterance === utterance && previous.text === text) {
    return;
  }
  captionHistory.push({ text, utterance });
  captionHistory = captionHistory.slice(-5);
  renderCaptionHistory();
}

function showTranslationCaption(text, revision, utterance, isFinal, code) {
  const normalized = typeof text === "string" ? text.trim() : "";
  if (!normalized || revision <= captionSuppressThrough) {
    return;
  }
  if (
    revision === captionRevision &&
    normalized === currentCaption &&
    isFinal === currentCaptionFinal
  ) {
    return;
  }

  const language = languages[code];
  captionLanguage.textContent = language
    ? `${language.name} translation`
    : "Live translation";
  captionCurrent.lang = captionLanguageTags[code] || "";
  captionPrevious.lang = captionLanguageTags[code] || "";

  if (utterance !== currentCaptionUtterance && currentCaption) {
    rememberCaption(currentCaption, currentCaptionUtterance);
  }

  currentCaption = normalized;
  currentCaptionUtterance = utterance;
  currentCaptionFinal = isFinal;
  captionRevision = revision;
  captionCurrent.textContent = normalized;
  captionCurrent.classList.remove("caption-placeholder");
  captionPhase.textContent = isFinal ? "Final" : "Streaming";
  translationCaptions.classList.remove("is-idle", "is-streaming", "is-final");
  translationCaptions.classList.add(isFinal ? "is-final" : "is-streaming");
  centerCurrentCaption();
}

function updateLanguageMap(code, language, announce) {
  let selectedMarker;

  languageOptions.forEach((marker) => {
    const selected = marker.dataset.code === code;
    marker.classList.toggle("is-selected", selected);
    marker.setAttribute("aria-pressed", String(selected));

    if (selected) {
      selectedMarker = marker;
    }
  });

  if (!selectedMarker) {
    return;
  }

  setFlagImage(mapFlag, language);
  mapCode.textContent = language.code;
  mapName.textContent = language.name;
  mapCountry.textContent = language.country;
  setFlagImage(globeLabelFlag, language);
  globeLabelCountry.textContent = language.country;
  languageMap.setAttribute(
    "aria-label",
    `${language.name} selected. The globe is focused on ${language.country}.`,
  );
  focusGlobeCountry(language);

  if (announce) {
    selectedMarker.classList.remove("is-changing");
    window.requestAnimationFrame(() =>
      selectedMarker.classList.add("is-changing"),
    );
  }
}

function latLonToVector(lat, lon, radius = 1) {
  const polar = THREE.MathUtils.degToRad(90 - lat);
  const azimuth = THREE.MathUtils.degToRad(lon + 180);

  return new THREE.Vector3(
    -radius * Math.cos(azimuth) * Math.sin(polar),
    radius * Math.cos(polar),
    radius * Math.sin(azimuth) * Math.sin(polar),
  );
}

function initGlobe() {
  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  try {
    const renderer = new THREE.WebGLRenderer({
      canvas: globeCanvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.set(0, 0, 3.7);

    const presentationGroup = new THREE.Group();
    const globeGroup = new THREE.Group();
    presentationGroup.add(globeGroup);
    scene.add(presentationGroup);

    scene.add(new THREE.HemisphereLight(0xeaf5ff, 0x102848, 2.25));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(2.8, 2.2, 4);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xff7a32, 1.35);
    rimLight.position.set(-3, 0.5, -2);
    scene.add(rimLight);

    const texture = new THREE.TextureLoader().load(
      "./assets/world-equirectangular.png",
      () => {
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
        texture.needsUpdate = true;
        languageMap.dataset.globeReady = "true";
      },
      undefined,
      () => languageMap.classList.add("has-texture-error"),
    );
    texture.colorSpace = THREE.SRGBColorSpace;

    const globeGeometry = new THREE.SphereGeometry(1, 128, 80);
    const globeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        landMask: { value: texture },
        time: { value: 0 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vViewNormal;
        varying vec3 vObjectNormal;

        void main() {
          vUv = uv;
          vViewNormal = normalize(normalMatrix * normal);
          vObjectNormal = normalize(normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D landMask;
        uniform float time;
        varying vec2 vUv;
        varying vec3 vViewNormal;
        varying vec3 vObjectNormal;

        float gridLine(float coordinate, float divisions, float width) {
          float cell = abs(fract(coordinate * divisions + 0.5) - 0.5) / fwidth(coordinate * divisions);
          return 1.0 - min(cell / width, 1.0);
        }

        float hash21(vec2 point) {
          point = fract(point * vec2(123.34, 456.21));
          point += dot(point, point + 45.32);
          return fract(point.x * point.y);
        }

        void main() {
          float rawLand = texture2D(landMask, vUv).a;
          float land = smoothstep(0.16, 0.72, rawLand);
          float coastline = smoothstep(0.012, 0.055, abs(dFdx(rawLand)) + abs(dFdy(rawLand)));
          float northLight = smoothstep(0.02, 0.98, vUv.y);
          float micro = hash21(floor(vUv * vec2(640.0, 320.0)));
          float wave = sin(vUv.x * 76.0 + vUv.y * 31.0 + time * 0.28) * 0.5 + 0.5;

          vec3 oceanDeep = vec3(0.76, 0.82, 0.86);
          vec3 oceanShelf = vec3(0.90, 0.92, 0.90);
          vec3 ocean = mix(oceanDeep, oceanShelf, northLight * 0.68 + wave * 0.055);

          vec3 landShadow = vec3(0.16, 0.28, 0.44);
          vec3 landIce = vec3(0.38, 0.52, 0.67);
          vec3 landBase = mix(landShadow, landIce, northLight * 0.78 + micro * 0.08);

          float longitude = gridLine(vUv.x, 24.0, 0.48);
          float latitude = gridLine(vUv.y, 12.0, 0.48);
          float grid = max(longitude, latitude);
          vec3 lightDirection = normalize(vec3(-0.42, 0.64, 1.0));
          float diffuse = 0.78 + 0.28 * max(dot(vViewNormal, lightDirection), 0.0);
          float fresnel = pow(1.0 - max(vViewNormal.z, 0.0), 3.15);
          float polarGlow = pow(abs(vObjectNormal.y), 7.0);

          vec3 color = mix(ocean, landBase, land);
          color *= diffuse;
          color += vec3(0.12, 0.31, 0.58) * grid * (0.07 + land * 0.04);
          color += vec3(0.18, 0.55, 0.86) * coastline * 0.32;
          color += vec3(0.28, 0.62, 0.88) * fresnel * 0.22;
          color += vec3(0.30, 0.48, 0.66) * polarGlow * 0.07;
          gl_FragColor = vec4(color, 1.0);
        }
      `,
      extensions: {
        derivatives: true,
      },
    });
    const globe = new THREE.Mesh(globeGeometry, globeMaterial);
    globeGroup.add(globe);

    const grid = new THREE.Mesh(
      new THREE.SphereGeometry(1.008, 36, 24),
      new THREE.MeshBasicMaterial({
        color: 0x8ecaff,
        transparent: true,
        opacity: 0.036,
        wireframe: true,
        depthWrite: false,
      }),
    );
    globeGroup.add(grid);

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.095, 96, 64),
      new THREE.ShaderMaterial({
        vertexShader: `
          varying vec3 vNormal;
          void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          varying vec3 vNormal;
          void main() {
            float facing = max(dot(vNormal, vec3(0.0, 0.0, 1.0)), 0.0);
            float rim = pow(1.0 - facing, 2.75);
            vec3 atmosphere = mix(vec3(0.12, 0.42, 0.78), vec3(0.52, 0.78, 0.96), rim);
            gl_FragColor = vec4(atmosphere, rim * 0.32);
          }
        `,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.BackSide,
        transparent: true,
      }),
    );
    globeGroup.add(atmosphere);

    const orbitGroup = new THREE.Group();
    const orbitSpecs = [
      { color: 0xff7a32, opacity: 0.34, radius: 1.22, tilt: [1.05, 0.45, 0.15] },
      { color: 0x5edcff, opacity: 0.24, radius: 1.29, tilt: [0.44, 1.04, -0.28] },
      { color: 0xbcdfff, opacity: 0.11, radius: 1.36, tilt: [1.42, -0.2, 0.58] },
    ];
    orbitSpecs.forEach((spec) => {
      const orbit = new THREE.Mesh(
        new THREE.TorusGeometry(spec.radius, 0.0045, 8, 220),
        new THREE.MeshBasicMaterial({
          color: spec.color,
          transparent: true,
          opacity: spec.opacity,
          depthWrite: false,
        }),
      );
      orbit.rotation.set(...spec.tilt);
      orbitGroup.add(orbit);
    });
    presentationGroup.add(orbitGroup);

    const particlePositions = [];
    for (let index = 0; index < 260; index += 1) {
      const radius = 1.25 + ((index * 37) % 87) / 100;
      const angle = index * 2.39996;
      const height = -1 + ((index * 53) % 101) / 50;
      const radial = Math.sqrt(Math.max(0, 1 - height * height));
      particlePositions.push(
        radius * radial * Math.cos(angle),
        radius * height,
        radius * radial * Math.sin(angle),
      );
    }
    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(particlePositions, 3),
    );
    const particles = new THREE.Points(
      particleGeometry,
      new THREE.PointsMaterial({
        color: 0x73dfff,
        size: 0.009,
        transparent: true,
        opacity: 0.38,
        depthWrite: false,
      }),
    );
    presentationGroup.add(particles);

    const singaporeVector = latLonToVector(1.3521, 103.8198).normalize();
    const originRoot = new THREE.Group();
    originRoot.position.copy(singaporeVector).multiplyScalar(1.027);
    originRoot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), singaporeVector);
    const originDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.029, 20, 14),
      new THREE.MeshBasicMaterial({ color: 0xd8fbff }),
    );
    const originHalo = new THREE.Mesh(
      new THREE.SphereGeometry(0.061, 24, 16),
      new THREE.MeshBasicMaterial({
        blending: THREE.AdditiveBlending,
        color: 0x4cdfff,
        depthWrite: false,
        opacity: 0.24,
        transparent: true,
      }),
    );
    const originRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.069, 0.007, 10, 48),
      new THREE.MeshBasicMaterial({
        color: 0x73f2ff,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
      }),
    );
    const originRingOuter = new THREE.Mesh(
      new THREE.TorusGeometry(0.112, 0.0035, 10, 64),
      new THREE.MeshBasicMaterial({
        blending: THREE.AdditiveBlending,
        color: 0x8ff8ff,
        depthWrite: false,
        opacity: 0.58,
        transparent: true,
      }),
    );
    originRoot.add(originHalo, originDot, originRing, originRingOuter);
    globeGroup.add(originRoot);

    const routeGeometry = new THREE.BufferGeometry();
    const routeMaterial = new THREE.LineDashedMaterial({
      color: 0xff9b5f,
      transparent: true,
      opacity: 0.94,
      dashSize: 0.045,
      gapSize: 0.025,
      depthWrite: false,
    });
    const routeGlowMaterial = new THREE.LineBasicMaterial({
      color: 0xff7a32,
      transparent: true,
      opacity: 0.2,
      depthWrite: false,
    });
    const routeLine = new THREE.Line(routeGeometry, routeMaterial);
    const routeGlow = new THREE.Line(routeGeometry, routeGlowMaterial);
    globeGroup.add(routeGlow, routeLine);

    const routePulse = new THREE.Mesh(
      new THREE.SphereGeometry(0.016, 14, 10),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    globeGroup.add(routePulse);
    let activeRouteCurve = null;

    const updateRoute = (targetVector) => {
      const routePoints = [];
      const pointCount = 96;
      for (let index = 0; index <= pointCount; index += 1) {
        const progress = index / pointCount;
        const point = singaporeVector
          .clone()
          .multiplyScalar(1 - progress)
          .add(targetVector.clone().multiplyScalar(progress))
          .normalize()
          .multiplyScalar(1.034 + Math.sin(Math.PI * progress) * 0.18);
        routePoints.push(point);
      }
      routeGeometry.setFromPoints(routePoints);
      routeLine.computeLineDistances();
      activeRouteCurve = new THREE.CatmullRomCurve3(routePoints);
    };

    const markerRoot = new THREE.Group();
    const markerDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.043, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0xff6b1f }),
    );
    const markerRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.085, 0.011, 12, 48),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
      }),
    );
    const markerRingOuter = new THREE.Mesh(
      new THREE.TorusGeometry(0.125, 0.004, 10, 64),
      new THREE.MeshBasicMaterial({
        color: 0x7fc5ff,
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      }),
    );
    const markerLight = new THREE.PointLight(0xff6b1f, 2.5, 0.8);
    markerLight.position.z = 0.1;
    markerRoot.add(markerDot, markerRing, markerRingOuter, markerLight);
    globeGroup.add(markerRoot);

    const front = new THREE.Vector3(0, 0, 1);
    const north = new THREE.Vector3(0, 1, 0);
    const markerNormal = new THREE.Vector3();
    const focusNormal = new THREE.Vector3();
    const focusRight = new THREE.Vector3();
    const focusUp = new THREE.Vector3();
    const focusBasis = new THREE.Matrix4();
    const targetQuaternion = new THREE.Quaternion();
    const yawQuaternion = new THREE.Quaternion();
    const yAxis = new THREE.Vector3(0, 1, 0);
    const markerWorld = new THREE.Vector3();
    const markerFacing = new THREE.Vector3();
    const projectedMarker = new THREE.Vector3();
    const originWorld = new THREE.Vector3();
    const originFacing = new THREE.Vector3();
    const projectedOrigin = new THREE.Vector3();
    const cameraDirection = camera.position.clone().normalize();
    let isDragging = false;
    let lastPointerX = 0;

    const resize = () => {
      const width = Math.max(1, globeViewport.clientWidth);
      const height = Math.max(1, globeViewport.clientHeight);
      const compact = width < 720;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.position.z = compact ? 6.9 : 3.7;
      presentationGroup.position.set(
        compact ? 0 : Math.min(0.52, width / 2200),
        compact ? -1.05 : -0.05,
        0,
      );
      presentationGroup.scale.setScalar(compact ? 1.08 : 1.08);
      camera.updateProjectionMatrix();
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(globeViewport);
    resize();

    const focusCountry = (language) => {
      markerNormal.copy(latLonToVector(language.lat, language.lon)).normalize();
      markerRoot.position.copy(markerNormal).multiplyScalar(1.035);
      markerRoot.quaternion.setFromUnitVectors(front, markerNormal);
      focusNormal
        .copy(markerNormal)
        .multiplyScalar(0.78)
        .addScaledVector(singaporeVector, 0.22)
        .normalize();
      focusRight.crossVectors(north, focusNormal).normalize();
      focusUp.crossVectors(focusNormal, focusRight).normalize();
      focusBasis.makeBasis(focusRight, focusUp, focusNormal);
      targetQuaternion.setFromRotationMatrix(focusBasis).invert();
      updateRoute(markerNormal);
      globeCountryLabel.classList.remove("is-visible");
      globeSingaporeLabel.classList.remove("is-visible");
    };

    globeViewport.addEventListener("pointerdown", (event) => {
      isDragging = true;
      lastPointerX = event.clientX;
      globeViewport.classList.add("is-dragging");
      globeViewport.setPointerCapture?.(event.pointerId);
    });

    globeViewport.addEventListener("pointermove", (event) => {
      if (!isDragging) {
        return;
      }

      const deltaX = event.clientX - lastPointerX;
      lastPointerX = event.clientX;
      yawQuaternion.setFromAxisAngle(yAxis, deltaX * 0.005);
      targetQuaternion.premultiply(yawQuaternion);
    });

    const endDrag = (event) => {
      isDragging = false;
      globeViewport.classList.remove("is-dragging");
      globeViewport.releasePointerCapture?.(event.pointerId);
    };

    globeViewport.addEventListener("pointerup", endDrag);
    globeViewport.addEventListener("pointercancel", endDrag);

    const animationStartedAt = window.performance.now();
    const render = () => {
      const elapsed = (window.performance.now() - animationStartedAt) / 1000;
      globeGroup.quaternion.slerp(
        targetQuaternion,
        reduceMotion ? 1 : isDragging ? 0.28 : 0.085,
      );
      presentationGroup.rotation.z = 0;
      orbitGroup.rotation.z = elapsed * 0.024;
      orbitGroup.rotation.y = Math.sin(elapsed * 0.22) * 0.045;
      particles.rotation.y = elapsed * 0.018;
      globeMaterial.uniforms.time.value = elapsed;
      if (activeRouteCurve) {
        routePulse.position.copy(
          activeRouteCurve.getPointAt((elapsed * 0.115) % 1),
        );
      }
      const markerScale = 1 + Math.sin(elapsed * 4.2) * 0.14;
      markerRing.scale.setScalar(markerScale);
      markerRingOuter.scale.setScalar(1 + Math.sin(elapsed * 2.8 + 0.8) * 0.1);
      markerRingOuter.material.opacity = 0.48 + Math.sin(elapsed * 3.1) * 0.18;
      originRing.scale.setScalar(1 + Math.sin(elapsed * 3.4 + 1.2) * 0.16);
      originRingOuter.scale.setScalar(
        1 + Math.sin(elapsed * 2.25 + 0.45) * 0.2,
      );
      originRingOuter.material.opacity =
        0.38 + Math.sin(elapsed * 2.7 + 0.8) * 0.18;
      originHalo.scale.setScalar(0.92 + Math.sin(elapsed * 3.1) * 0.16);

      scene.updateMatrixWorld();
      markerRoot.getWorldPosition(markerWorld);
      projectedMarker.copy(markerWorld).project(camera);
      originRoot.getWorldPosition(originWorld);
      projectedOrigin.copy(originWorld).project(camera);
      const width = globeViewport.clientWidth;
      const height = globeViewport.clientHeight;
      const facingCamera =
        markerFacing.copy(markerWorld).normalize().dot(cameraDirection) > 0.08;
      const originFacingCamera =
        originFacing.copy(originWorld).normalize().dot(cameraDirection) > 0.04;

      if (facingCamera && !isDragging) {
        const labelX = THREE.MathUtils.clamp(
          (projectedMarker.x * 0.5 + 0.5) * width,
          72,
          width - 72,
        );
        const labelY = THREE.MathUtils.clamp(
          (-projectedMarker.y * 0.5 + 0.5) * height - 30,
          35,
          height - 52,
        );
        globeCountryLabel.style.left = `${labelX}px`;
        globeCountryLabel.style.top = `${labelY}px`;
        globeCountryLabel.classList.add("is-visible");
      } else {
        globeCountryLabel.classList.remove("is-visible");
      }

      if (originFacingCamera && !isDragging) {
        const originLabelX = THREE.MathUtils.clamp(
          (projectedOrigin.x * 0.5 + 0.5) * width,
          88,
          width - 88,
        );
        const originLabelY = THREE.MathUtils.clamp(
          (-projectedOrigin.y * 0.5 + 0.5) * height,
          58,
          height - 62,
        );
        globeSingaporeLabel.style.left = `${originLabelX}px`;
        globeSingaporeLabel.style.top = `${originLabelY}px`;
        globeSingaporeLabel.classList.add("is-visible");
      } else {
        globeSingaporeLabel.classList.remove("is-visible");
      }

      renderer.render(scene, camera);
    };

    renderer.setAnimationLoop(render);
    languageMap.dataset.globeRenderer = "webgl";
    languageMap.dataset.globeReady = "loading";
    return focusCountry;
  } catch (error) {
    languageMap.classList.add("is-fallback");
    languageMap.dataset.globeRenderer = "fallback";
    console.warn("Globe renderer unavailable", error);
    return () => {};
  }
}

function createWaveform() {
  const waveform = document.querySelector(".wave-bars-primary");
  if (!waveform) {
    return;
  }

  const heights = [
    12, 20, 15, 31, 24, 42, 30, 51, 35, 63, 44, 76, 56, 92, 68, 100, 74,
    56, 83, 51, 70, 42, 59, 35, 48, 29, 39, 23, 32, 18, 25,
  ];

  const fragment = document.createDocumentFragment();
  heights.forEach((height, index) => {
    const bar = document.createElement("i");
    bar.style.setProperty("--bar-height", height);
    bar.style.setProperty("--bar-index", index);
    bar.style.setProperty("--wave-duration", `${820 + ((index * 83) % 620)}ms`);
    fragment.appendChild(bar);
  });
  waveform.appendChild(fragment);
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => {
    toast.classList.remove("is-visible");
  }, 2400);
}

function setLanguage(code, announce = true) {
  const language = languages[code];
  if (!language) {
    return;
  }

  targetLanguage.value = code;
  setFlagImage(selectFlag, language);
  selectCode.textContent = language.code;
  updateLanguageMap(code, language, announce);

  if (announce) {
    window.clearTimeout(selectionTimer);
    selectShell.classList.add("is-changing");
    selectionTimer = window.setTimeout(() => {
      selectShell.classList.remove("is-changing");
      languageOptions.forEach((option) =>
        option.classList.remove("is-changing"),
      );
    }, 520);
    showToast(`Target language set to ${language.name}`);
  }
}

function initMotion() {
  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  const revealSections = [
    document.querySelector(".translation-surface"),
  ].filter(Boolean);

  document.body.classList.add("motion-prep");
  languageOptions.forEach((option, index) => {
    option.style.setProperty("--reveal-index", index);
  });
  revealSections.forEach((section) =>
    section.classList.add("reveal-on-scroll"),
  );

  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealSections.forEach((section) => section.classList.add("is-visible"));
  } else {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) {
            return;
          }
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.12 },
    );
    revealSections.forEach((section) => observer.observe(section));
    window.setTimeout(() => {
      revealSections.forEach((section) => section.classList.add("is-visible"));
    }, 1100);
  }

  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      document.body.classList.add("is-mounted");
    });
  });

  if (
    reduceMotion ||
    !window.matchMedia("(hover: hover) and (pointer: fine)").matches
  ) {
    return;
  }

  let pointerFrame = 0;
  let pointerX = 0;
  let pointerY = 0;

  culturalHero.addEventListener("pointermove", (event) => {
    const bounds = culturalHero.getBoundingClientRect();
    pointerX = (event.clientX - bounds.left) / bounds.width - 0.5;
    pointerY = (event.clientY - bounds.top) / bounds.height - 0.5;

    if (pointerFrame) {
      return;
    }

    pointerFrame = window.requestAnimationFrame(() => {
      culturalHero.style.setProperty("--product-x", `${pointerX * 14}px`);
      culturalHero.style.setProperty("--product-y", `${pointerY * 10}px`);
      culturalHero.style.setProperty("--backdrop-x", `${pointerX * -7}px`);
      culturalHero.style.setProperty("--backdrop-y", `${pointerY * -5}px`);
      pointerFrame = 0;
    });
  });

  culturalHero.addEventListener("pointerleave", () => {
    culturalHero.style.setProperty("--product-x", "0px");
    culturalHero.style.setProperty("--product-y", "0px");
    culturalHero.style.setProperty("--backdrop-x", "0px");
    culturalHero.style.setProperty("--backdrop-y", "0px");
  });
}

function setPaused(paused, announce = true) {
  isPaused = paused;
  document.body.classList.toggle("is-paused", paused);
  listenControl.setAttribute("aria-pressed", String(paused));
  listenControl.setAttribute(
    "aria-label",
    paused ? "Resume translation" : "Pause translation",
  );
  listenButtonLabel.textContent = paused ? "Resume" : "Pause";
  listeningCopy.textContent = paused
    ? "Translation paused"
    : "Listening for any language...";
  if (announce) {
    showToast(paused ? "Translation paused" : "Translation resumed");
  }
}

function setBluetoothConnected(connected) {
  if (!bluetoothLensStatus) {
    return;
  }

  const isConnected = Boolean(connected);

  bluetoothLensStatus.dataset.connected = String(isConnected);
  bluetoothLensStatus.classList.toggle("is-connected", isConnected);
  bluetoothLensStatus.setAttribute(
    "aria-label",
    isConnected ? "Bluetooth connected" : "Bluetooth disconnected",
  );
}

function setRuntimeOnline(online) {
  runtimeOnline = Boolean(online);
  runtimeStatus?.classList.toggle("is-offline", !runtimeOnline);
  if (runtimeStatusText) {
    runtimeStatusText.textContent = runtimeOnline
      ? "Translation online"
      : "Runtime offline";
  }
  setBluetoothConnected(runtimeOnline);
}

async function postControl(payload) {
  const response = await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(result.error || `control request failed (${response.status})`);
  }
  return result;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

async function postPlayback(enabled, nodeName, volume) {
  const response = await fetchWithTimeout("/api/playback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled, nodeName, volume }),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(result.error || `playback request failed (${response.status})`);
  }
  return result;
}

function applyPlaybackStatus(status) {
  const outputs = Array.isArray(status.outputs) ? status.outputs : [];
  const previousSelection = speakerOutput.value;
  const requestedSelection = status.nodeName || previousSelection;
  const optionModels = outputs.map((output) => ({
    value: output.nodeName,
    label: output.name,
  }));

  if (
    requestedSelection &&
    !outputs.some((output) => output.nodeName === requestedSelection)
  ) {
    optionModels.unshift({
      value: requestedSelection,
      label: "Selected speaker unavailable",
    });
  }
  if (optionModels.length === 0) {
    optionModels.push({ value: "", label: "No speaker detected" });
  }
  const optionsSignature = JSON.stringify(optionModels);
  if (optionsSignature !== playbackOutputsSignature) {
    const options = optionModels.map(({ value, label }) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    });
    speakerOutput.replaceChildren(...options);
    playbackOutputsSignature = optionsSignature;
  }
  speakerOutput.value =
    optionModels.some((option) => option.value === requestedSelection)
      ? requestedSelection
      : optionModels[0].value;

  const enabled = Boolean(status.enabled);
  const active = Boolean(status.active);
  const error = typeof status.error === "string" ? status.error : "";
  const volume = Number.isFinite(status.volume)
    ? Math.round(Math.min(1, Math.max(0, status.volume)) * 100)
    : 90;
  if (!playbackRequestActive) {
    speakerMirror.checked = enabled;
  }
  const hasConfiguredOutput = Boolean(speakerOutput.value);
  speakerMirror.disabled = !hasConfiguredOutput;
  speakerOutput.disabled = outputs.length === 0;
  if (!speakerVolumeEditing && !playbackRequestActive) {
    speakerVolume.value = String(volume);
    speakerVolumeValue.textContent = `${volume}%`;
    speakerVolume.style.setProperty("--speaker-volume", `${volume}%`);
  }
  speakerVolume.disabled = !hasConfiguredOutput;
  speakerMirrorControl.classList.toggle("is-active", active);
  speakerMirrorControl.classList.toggle("has-error", Boolean(error));

  if (error) {
    speakerMirrorStatus.textContent = error;
  } else if (active) {
    speakerMirrorStatus.textContent = "Glasses + speaker live";
  } else if (enabled) {
    speakerMirrorStatus.textContent = "Waiting for translation audio";
  } else if (outputs.length > 0) {
    speakerMirrorStatus.textContent = "Glasses only · speaker ready";
  } else {
    speakerMirrorStatus.textContent = "Glasses only · connect a speaker";
  }
}

async function refreshPlaybackStatus() {
  if (playbackRequestActive) {
    return;
  }
  const requestedAtRevision = playbackDesiredRevision;
  try {
    const response = await fetchWithTimeout(
      "/api/playback",
      { cache: "no-store" },
      3500,
    );
    if (!response.ok) {
      throw new Error(`playback status failed (${response.status})`);
    }
    const status = await response.json();
    if (
      !playbackRequestActive &&
      requestedAtRevision === playbackDesiredRevision
    ) {
      applyPlaybackStatus(status);
    }
  } catch (error) {
    speakerMirrorControl.classList.add("has-error");
    speakerMirrorStatus.textContent = "Playback control unavailable";
    console.debug("StreamGlass playback control unavailable", error);
  }
}

function queuePlaybackUpdate(toastMessage = "") {
  if (speakerMirror.checked && !speakerOutput.value) {
    speakerMirror.checked = false;
    showToast("Connect a speaker output first");
    return;
  }
  playbackDesiredRevision += 1;
  playbackRequestPending = true;
  if (toastMessage) {
    playbackQueuedToast = toastMessage;
  }
  void flushPlaybackUpdate();
}

async function flushPlaybackUpdate() {
  if (playbackRequestActive || !playbackRequestPending) {
    return;
  }
  playbackRequestPending = false;
  const revision = playbackDesiredRevision;
  const enabled = speakerMirror.checked;
  const nodeName = speakerOutput.value;
  const volume = Number(speakerVolume.value) / 100;
  const toastMessage = playbackQueuedToast;
  playbackQueuedToast = "";
  playbackRequestActive = true;
  speakerMirrorControl.classList.add("is-updating");
  try {
    const status = await postPlayback(enabled, nodeName, volume);
    if (revision === playbackDesiredRevision) {
      applyPlaybackStatus(status);
      if (toastMessage) {
        showToast(toastMessage);
      }
    }
  } catch (error) {
    if (revision === playbackDesiredRevision) {
      showToast(`Playback change failed: ${error.message}`);
    }
  } finally {
    playbackRequestActive = false;
    speakerMirrorControl.classList.remove("is-updating");
    if (playbackRequestPending) {
      void flushPlaybackUpdate();
    } else {
      void refreshPlaybackStatus();
    }
  }
}

function requestSpeakerMirror(enabled) {
  queuePlaybackUpdate(
    enabled ? "Speaker mirror enabled" : "Glasses-only playback",
  );
}

function requestSpeakerVolume() {
  queuePlaybackUpdate();
}

function scheduleSpeakerVolume() {
  speakerVolumeEditing = true;
  const volume = Number(speakerVolume.value);
  speakerVolumeValue.textContent = `${volume}%`;
  speakerVolume.style.setProperty("--speaker-volume", `${volume}%`);
  window.clearTimeout(speakerVolumeRequestTimer);
  speakerVolumeRequestTimer = window.setTimeout(
    () => requestSpeakerVolume(),
    220,
  );
}

function applyRuntimeStatus(status) {
  setRuntimeOnline(status.online === true);
  if (!status.online) {
    return;
  }

  setPaused(Boolean(status.paused), false);
  const target = status.languagePending
    ? status.requestedLanguage
    : status.targetLanguage;
  if (languages[target] && targetLanguage.value !== target) {
    setLanguage(target, false);
  }
  if (!status.languagePending && languages[status.targetLanguage]) {
    confirmedLanguage = status.targetLanguage;
  }

  const processedUtterances = Number.isSafeInteger(status.processedUtterances)
    ? status.processedUtterances
    : Number.parseInt(status.processedUtterances, 10) || 0;
  const translationRevision = Number.isSafeInteger(status.translationRevision)
    ? status.translationRevision
    : Number.parseInt(status.translationRevision, 10) || processedUtterances;
  const translationUtterance = Number.isSafeInteger(status.translationUtterance)
    ? status.translationUtterance
    : Number.parseInt(status.translationUtterance, 10) || processedUtterances;
  const translationFinal = status.translationFinal !== false;
  const appliedCaptionLanguage = languages[status.targetLanguage]
    ? status.targetLanguage
    : confirmedLanguage;

  if (
    processedUtterances < lastProcessedUtterances ||
    translationRevision < lastTranslationRevision
  ) {
    clearTranslationCaptions(appliedCaptionLanguage);
  } else if (
    !status.languagePending &&
    appliedCaptionLanguage !== captionLanguageCode
  ) {
    clearTranslationCaptions(
      appliedCaptionLanguage,
      lastTranslationRevision,
    );
  }
  lastProcessedUtterances = processedUtterances;
  lastTranslationRevision = translationRevision;

  if (!status.languagePending) {
    const latestTranslation =
      typeof status.latestTranslation === "string"
        ? status.latestTranslation.trim()
        : "";
    showTranslationCaption(
      latestTranslation,
      translationRevision,
      translationUtterance,
      translationFinal,
      appliedCaptionLanguage,
    );
  }
  if (status.lastError) {
    console.warn("StreamGlass runtime control:", status.lastError);
  }
}

async function refreshRuntimeStatus() {
  if (statusRequestActive) {
    return;
  }
  statusRequestActive = true;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status request failed (${response.status})`);
    }
    applyRuntimeStatus(await response.json());
  } catch (error) {
    setRuntimeOnline(false);
    console.debug("StreamGlass runtime unavailable", error);
  } finally {
    statusRequestActive = false;
  }
}

function scheduleRuntimeFallback() {
  if (runtimeFallbackTimer) {
    return;
  }
  runtimeFallbackTimer = window.setTimeout(async () => {
    runtimeFallbackTimer = undefined;
    await refreshRuntimeStatus();
    if (runtimeEvents && runtimeEvents.readyState !== EventSource.OPEN) {
      scheduleRuntimeFallback();
    }
  }, 500);
}

function connectRuntimeEvents() {
  if (!("EventSource" in window)) {
    window.setInterval(() => void refreshRuntimeStatus(), 250);
    return;
  }

  runtimeEvents = new EventSource("/api/events");
  runtimeEvents.addEventListener("runtime", (event) => {
    try {
      applyRuntimeStatus(JSON.parse(event.data));
    } catch (error) {
      console.warn("Invalid StreamGlass runtime event", error);
    }
  });
  runtimeEvents.addEventListener("open", () => {
    window.clearTimeout(runtimeFallbackTimer);
    runtimeFallbackTimer = undefined;
  });
  runtimeEvents.addEventListener("error", () => {
    scheduleRuntimeFallback();
  });
}

async function requestLanguage(code) {
  const previous = confirmedLanguage;
  if (code !== captionLanguageCode) {
    clearTranslationCaptions(code, lastTranslationRevision);
  }
  setLanguage(code);
  try {
    const status = await postControl({ action: "language", language: code });
    applyRuntimeStatus(status);
  } catch (error) {
    setLanguage(previous, false);
    setRuntimeOnline(false);
    showToast(`Language change failed: ${error.message}`);
  }
}

async function requestPaused(paused) {
  const previous = isPaused;
  setPaused(paused);
  try {
    const status = await postControl({
      action: paused ? "pause" : "resume",
    });
    applyRuntimeStatus(status);
  } catch (error) {
    setPaused(previous, false);
    setRuntimeOnline(false);
    showToast(`Runtime control failed: ${error.message}`);
  }
}

languageMap.addEventListener("click", (event) => {
  const option = event.target.closest(".globe-target");
  if (!option) {
    return;
  }

  void requestLanguage(option.dataset.code);
});

targetLanguage.addEventListener("change", () => {
  void requestLanguage(targetLanguage.value);
});

listenControl.addEventListener("click", () => {
  void requestPaused(!isPaused);
});

speakerMirror.addEventListener("change", () => {
  void requestSpeakerMirror(speakerMirror.checked);
});

speakerOutput.addEventListener("change", () => {
  void requestSpeakerMirror(speakerMirror.checked);
});

speakerVolume.addEventListener("input", scheduleSpeakerVolume);
speakerVolume.addEventListener("change", () => {
  speakerVolumeEditing = false;
  window.clearTimeout(speakerVolumeRequestTimer);
  requestSpeakerVolume();
});
speakerVolume.addEventListener("pointerdown", () => {
  speakerVolumeEditing = true;
});
speakerVolume.addEventListener("pointercancel", () => {
  speakerVolumeEditing = false;
});
speakerVolume.addEventListener("blur", () => {
  speakerVolumeEditing = false;
});

focusGlobeCountry = initGlobe();
initMotion();
createWaveform();
setLanguage("eng", false);
setRuntimeOnline(false);
void refreshRuntimeStatus();
void refreshPlaybackStatus();
connectRuntimeEvents();
window.setInterval(() => void refreshPlaybackStatus(), 2000);

window.setBluetoothConnected = setBluetoothConnected;
window.applyStreamGlassRuntimeStatus = applyRuntimeStatus;
