/* Lecture d'un titre foncier : dépôt du document, extraction, fiche à valider.
 *
 * Deux règles tiennent toute cette page :
 *   1. un champ prérempli qui cache son incertitude est pire qu'un champ vide.
 *      Toute valeur sous le seuil de relecture se voit, et on peut déplier ce que
 *      chaque lecteur a lu ;
 *   2. le bouton d'enregistrement obéit au verdict de l'API, jamais à cette page.
 *      C'est `jugement.entre_en_base` qui décide, pas le navigateur.
 */

const API = '';                 // même origine : l'API sert cette page
const SEUIL = 0.85;             // app/core/config.py — review_threshold

/* Le libellé de la case imprimée sur le document, pas le nom du champ dans le code. */
const GROUPES = [
  ['Identification du titre', [
    ['title_number',      'Titre n°'],
    ['mother_title',      'Titre mère n°'],
    ['lot',               'Lot n°'],
    ['subdivision',       'Lotissement'],
    ['requisition',       'Réquisition n°'],
    ['land_registry',     'Livre foncier'],
    ['nicad',             'NICAD'],
  ]],
  ['Localisation', [
    ['region',            'Région'],
    ['department',        'Département'],
    ['arrondissement',    'Arrondissement'],
    ['commune',           'Commune'],
  ]],
  ['Requérant', [
    ['owner',             'Requérant'],
  ]],
  ['Plan', [
    ['coordinate_system', 'Système de coord.'],
    ['scale',             'Échelle'],
    ['certified_on',      'Certifié le'],
    ['disclaimer',        'Mention portée'],
  ]],
];

const CARDINAUX = new Set(['title_number', 'nicad', 'commune', 'owner', 'printed_area']);

let extraction = null;          // l'objet rendu par l'API, édité sur place par l'agent

/* ------------------------------------------------------------------ dépôt */

const depot   = document.getElementById('depot');
const fichier = document.getElementById('fichier');

depot.addEventListener('click', () => fichier.click());
depot.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fichier.click(); }
});
['dragenter', 'dragover'].forEach((t) => depot.addEventListener(t, (e) => {
  e.preventDefault(); depot.classList.add('survol');
}));
['dragleave', 'drop'].forEach((t) => depot.addEventListener(t, (e) => {
  e.preventDefault(); depot.classList.remove('survol');
}));
depot.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) lire(f);
});
fichier.addEventListener('change', () => {
  if (fichier.files[0]) lire(fichier.files[0]);
});

/* --------------------------------------------------------------- lecture */

async function lire(f) {
  montrerSource(f);
  basculer({ attente: true });
  vider('messages');

  const corps = new FormData();
  corps.append('fichier', f);

  try {
    const r = await fetch(`${API}/documents?apercu=true`, { method: 'POST', body: corps });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `Erreur ${r.status}`);

    extraction = data.extraction;
    basculer({ fiche: true });
    poserVerdict(data.jugement);
    construireFiche();
    poserTrace(data.apercu_png);
    if (data.geometrie_refusee) {
      message(`Le polygone n'a pas pu être construit : ${data.geometrie_refusee}`, 'erreur');
    }
  } catch (err) {
    basculer({ vide: true });
    message(`Lecture impossible : ${err.message}`, 'erreur');
  }
}

function montrerSource(f) {
  depot.classList.add('replie');
  depot.querySelector('h2').textContent = 'Remplacer le document';
  const fig = document.getElementById('piece-source');
  const img = document.getElementById('apercu-source');
  if (f.type === 'application/pdf') {
    img.hidden = true;
  } else {
    img.hidden = false;
    img.src = URL.createObjectURL(f);
  }
  document.getElementById('legende-source').textContent =
    `${f.name} — ${(f.size / 1024 / 1024).toFixed(1)} Mo`;
  fig.hidden = false;
  document.getElementById('piece-trace').hidden = true;
}

function poserTrace(b64) {
  const fig = document.getElementById('piece-trace');
  if (!b64) { fig.hidden = true; return; }
  document.getElementById('apercu-trace').src = `data:image/png;base64,${b64}`;
  fig.hidden = false;
}

/* --------------------------------------------------------------- verdict */

function poserVerdict(j) {
  const titres = {
    exploitable: 'Exploitable — la parcelle peut entrer en base',
    a_relire:    'À relire — une relecture est nécessaire avant enregistrement',
    refuse:      'Refusé — le document ne peut pas être enregistré en l\'état',
  };
  document.getElementById('verdict').innerHTML = `
    <div class="verdict verdict--${j.decision}">
      <h2>${titres[j.decision] || j.decision}</h2>
      ${j.raisons.length ? `<ul>${j.raisons.map((r) => `<li>${txt(r)}</li>`).join('')}</ul>` : ''}
    </div>`;
  document.getElementById('btn-enregistrer').disabled = !j.entre_en_base;
  document.getElementById('raison-bloquee').hidden = j.entre_en_base;
}

/* ----------------------------------------------------------------- fiche */

function construireFiche() {
  const hote = document.getElementById('groupes');
  hote.innerHTML = '';

  for (const [titre, champs] of GROUPES) {
    const g = bloc(titre);
    for (const [nom, libelle] of champs) g.append(champTexte(nom, libelle));
    hote.append(g);
  }

  hote.append(blocSuperficie());
  hote.append(blocBornage());
  if (extraction.issues?.length) hote.append(blocAnomalies());
}

function bloc(titre) {
  const d = document.createElement('div');
  d.className = 'groupe';
  d.innerHTML = `<h3>${titre}</h3>`;
  return d;
}

/* Un champ : la valeur, sa confiance, et ce que chaque lecteur a lu. */
function champTexte(nom, libelle) {
  const f = extraction[nom] || {};
  const doute = aDouter(f, nom);

  const d = document.createElement('div');
  d.className = 'champ' + (doute ? ' champ--doute' : '');
  d.innerHTML = `
    <label for="c-${nom}">${libelle}</label>
    <input id="c-${nom}" value="${attr(f.value)}"
           placeholder="${f.value === null || f.value === undefined ? 'case non lue' : ''}">
    <span class="confiance">${confiance(f)}</span>`;

  const input = d.querySelector('input');
  input.addEventListener('change', () => {
    corriger(extraction[nom], input.value.trim() || null);
    d.classList.remove('champ--doute');
    d.querySelector('.confiance').textContent = 'corrigé';
  });

  const lectures = blocLectures(f);
  if (lectures) d.append(lectures);
  return d;
}

/* Les lectures brutes : c'est ce qui permet à l'agent de trancher lui-même. */
function blocLectures(f) {
  const lues = (f.readings || []).filter((l) => l.raw_text || l.normalized);
  if (lues.length < 1) return null;
  const det = document.createElement('details');
  det.className = 'lectures';
  det.innerHTML = `
    <summary>${lues.length} lecture${lues.length > 1 ? 's' : ''}</summary>
    <dl>${lues.map((l) => `
      <dt>${txt(l.source)}</dt>
      <dd>${txt(l.raw_text ?? l.normalized)}${
        l.confidence != null ? ` <span class="confiance">${pct(l.confidence)}</span>` : ''}</dd>
    `).join('')}</dl>`;
  return det;
}

function blocSuperficie() {
  const f = extraction.printed_area || {};
  const a = f.value || { ha: 0, a: 0, ca: 0 };
  const doute = aDouter(f, 'printed_area');

  const g = bloc('Superficie imprimée');
  const d = document.createElement('div');
  d.className = 'champ' + (doute ? ' champ--doute' : '');
  d.innerHTML = `
    <label>ha, a, ca</label>
    <span>
      <input class="nombre" id="c-ha" inputmode="numeric" value="${attr(a.ha)}" style="width:6ch">
      <input class="nombre" id="c-a"  inputmode="numeric" value="${attr(a.a)}"  style="width:5ch">
      <input class="nombre" id="c-ca" inputmode="numeric" value="${attr(a.ca)}" style="width:5ch">
      <span class="confiance" id="m2">${m2(a)}</span>
    </span>
    <span class="confiance">${confiance(f)}</span>`;

  d.querySelectorAll('input').forEach((input) => input.addEventListener('change', () => {
    const v = {
      ha: entier(d.querySelector('#c-ha').value),
      a:  entier(d.querySelector('#c-a').value),
      ca: entier(d.querySelector('#c-ca').value),
    };
    corriger(extraction.printed_area, v);
    d.querySelector('#m2').textContent = m2(v);
    d.classList.remove('champ--doute');
  }));

  const l = blocLectures(f);
  if (l) d.append(l);
  g.append(d);
  return g;
}

/* Le tableau de bornage, dans l'ordre imprimé. Les chiffres sont en chasse fixe :
 * des décimales alignées font sauter aux yeux une valeur mal lue. */
function blocBornage() {
  const g = bloc(`Bornage — ${extraction.points.length} sommet${
    extraction.points.length > 1 ? 's' : ''}`);

  if (!extraction.points.length) {
    g.insertAdjacentHTML('beforeend',
      '<p class="vide">Aucun tableau de coordonnées n\'a été lu sur ce document.</p>');
    return g;
  }

  const t = document.createElement('table');
  t.className = 'bornage';
  t.innerHTML = `<thead><tr>
      <th>Point</th><th>X (m)</th><th>Y (m)</th><th>Confiance</th>
    </tr></thead><tbody></tbody>`;
  const corps = t.querySelector('tbody');

  extraction.points.forEach((p, i) => {
    const doute = !fiable(p.x) || !fiable(p.y);
    const tr = document.createElement('tr');
    if (doute) tr.className = 'doute';
    tr.innerHTML = `
      <td>${txt(p.label)}</td>
      <td class="nombre"><input class="nombre" value="${coord(p.x)}" style="width:13ch"></td>
      <td class="nombre"><input class="nombre" value="${coord(p.y)}" style="width:13ch"></td>
      <td class="nombre confiance">${pct(Math.min(p.x.confidence, p.y.confidence))}</td>`;

    const [ix, iy] = tr.querySelectorAll('input');
    ix.addEventListener('change', () => {
      corriger(extraction.points[i].x, reel(ix.value)); tr.classList.remove('doute');
    });
    iy.addEventListener('change', () => {
      corriger(extraction.points[i].y, reel(iy.value)); tr.classList.remove('doute');
    });
    corps.append(tr);
  });

  g.append(t);
  return g;
}

/* Les anomalies, telles que l'API les constate. Un recouvrement est un fait
 * géométrique : le qualifier relève de l'humain et du droit. */
function blocAnomalies() {
  const g = bloc('Anomalies constatées');
  const rang = { error: 0, warning: 1, info: 2 };
  const liste = [...extraction.issues].sort((a, b) => rang[a.severity] - rang[b.severity]);
  g.insertAdjacentHTML('beforeend', liste.map((i) => `
    <div class="message ${i.severity === 'error' ? 'message--erreur' : ''}">
      ${txt(i.message)}${
        i.point_labels?.length ? ` <span class="confiance">(${
          i.point_labels.map(txt).join(', ')})</span>` : ''}
      <span class="reference">${txt(i.code)}</span>
    </div>`).join(''));
  return g;
}

/* ------------------------------------------------------------- commandes */

document.getElementById('btn-confronter').addEventListener('click', async (e) => {
  const b = e.currentTarget;
  b.disabled = true; b.textContent = 'Confrontation…';
  try {
    const r = await fetch(`${API}/parcelles/confrontation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ extraction }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || `Erreur ${r.status}`);

    if (!d.recouvrements.length) {
      message(`Aucune parcelle enregistrée ne recouvre celle-ci. `
        + `Surface calculée : ${Math.round(d.parcelle.aire_m2).toLocaleString('fr-FR')} m²`
        + `${d.parcelle.ecart_pct != null
            ? `, soit ${d.parcelle.ecart_pct} % d'écart avec la superficie imprimée` : ''}.`,
        'succes');
    } else {
      message(`${d.recouvrements.length} recouvrement${d.recouvrements.length > 1 ? 's' : ''} `
        + `détecté${d.recouvrements.length > 1 ? 's' : ''}, dont ${d.conflits} à instruire :`
        + `<ul>${d.recouvrements.map((x) =>
            `<li>${txt(x.b)} — ${txt(x.message)}</li>`).join('')}</ul>`,
        d.conflits ? 'erreur' : '');
    }
    // Le GeoJSON de confrontation porte la parcelle ET ses voisines d'emprise ;
    // les voisines sont déjà en base, seule la nouvelle est à superposer.
    // Deux parcelles d'un même propriétaire sont le cas le plus banal du cadastre :
    // on les montre ensemble, on ne conclut rien.
    if (d.rapprochements?.length) {
      message(`Le propriétaire s'écrit presque pareil sur `
        + `${d.rapprochements.length} titre${d.rapprochements.length > 1 ? 's' : ''} `
        + `déjà en base :<ul>${d.rapprochements.map((x) =>
            `<li>${txt(x.message)}</li>`).join('')}</ul>`);
    }

    sessionStorage.setItem('cadastre.confrontation', JSON.stringify({
      document: d.parcelle.document_id, geojson: d.geojson }));
    message('Les parcelles confrontées sont reportées sur la <a href="carte.html">carte</a>.');
  } catch (err) {
    message(`Confrontation impossible : ${err.message}`, 'erreur');
  } finally {
    b.disabled = false; b.textContent = 'Confronter à la base';
  }
});

document.getElementById('btn-enregistrer').addEventListener('click', async (e) => {
  const b = e.currentTarget;
  b.disabled = true; b.textContent = 'Enregistrement…';
  try {
    const r = await fetch(`${API}/parcelles`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ extraction }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || `Erreur ${r.status}`);
    message(`Parcelle ${txt(d.document_id)} enregistrée. `
      + `Elle apparaît désormais sur la <a href="carte.html">carte</a>.`, 'succes');
    b.textContent = 'Enregistrée';
  } catch (err) {
    message(`Enregistrement refusé : ${err.message}`, 'erreur');
    b.disabled = false; b.textContent = 'Enregistrer la parcelle';
  }
});

/* ----------------------------------------------------------------- outils */

/* Une valeur saisie par un agent est une valeur manuelle et certaine : c'est ce qui
 * la sort de la liste des champs à relire, et c'est tracé par `source`. */
function corriger(champ, valeur) {
  if (!champ) return;
  champ.value = valeur;
  champ.source = 'manual';
  champ.confidence = valeur === null ? 0 : 1;
}

function aDouter(f, nom) {
  if (f.value === null || f.value === undefined) return CARDINAUX.has(nom);
  return f.confidence < SEUIL;
}
const fiable = (f) => f.value !== null && f.value !== undefined && f.confidence >= SEUIL;

function confiance(f) {
  if (f.value === null || f.value === undefined) return 'non lu';
  return pct(f.confidence);
}
const pct = (c) => (c == null ? '' : `${Math.round(c * 100)} %`);
const m2  = (a) => `${(a.ha * 10000 + a.a * 100 + a.ca).toLocaleString('fr-FR')} m²`;

const entier = (v) => { const n = parseInt(v, 10); return Number.isFinite(n) ? n : 0; };

/* Le JSON rend 234773.6 là où le document imprime 234773.60 : le zéro final
 * disparaît dans le nombre, pas dans la lecture. On le réaffiche — mais on
 * n'en ajoute jamais. `raw_text` est la chaîne telle que lue sur le papier ;
 * c'est elle qui dit combien de décimales le document porte, deux ici, trois
 * ailleurs. Afficher un nombre fixe de décimales reviendrait à combler le trou. */
const decimales = (s) => {
  const m = String(s ?? '').replace(',', '.').match(/\.(\d+)\s*$/);
  return m ? m[1].length : null;
};
const coord = (f) => {
  if (!f || f.value === null || f.value === undefined) return '';
  const d = decimales(f.raw_text) ?? decimales(f.value) ?? 0;
  return Number(f.value).toFixed(d);
};
const reel   = (v) => { const n = parseFloat(String(v).replace(',', '.'));
                        return Number.isFinite(n) ? n : null; };

/* Tout ce qui vient de l'API traverse le document : on l'échappe. */
const txt  = (v) => String(v ?? '').replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const attr = (v) => (v === null || v === undefined ? '' : txt(v));

function basculer({ vide, attente, fiche }) {
  document.getElementById('fiche-vide').hidden = !vide;
  document.getElementById('attente').hidden    = !attente;
  document.getElementById('fiche').hidden      = !fiche;
}
function vider(id) { document.getElementById(id).innerHTML = ''; }

function message(html, ton = '') {
  const d = document.createElement('div');
  d.className = 'message' + (ton ? ` message--${ton}` : '');
  d.innerHTML = html;
  document.getElementById('messages').append(d);
}

/* L'état des lecteurs : si un modèle est tombé, l'agent doit le savoir avant de
 * déposer un document, pas après. La forme vient de GET /health/lecteurs. */
const NOMS = { paddle_vl: 'PaddleOCR-VL', qwen_vl: 'Qwen3-VL' };

(async () => {
  const e = document.getElementById('etat-lecteurs');
  try {
    const r = await fetch(`${API}/health/lecteurs`);
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    const absents = Object.entries(d.lecteurs)
      .filter(([, v]) => !v.joignable)
      .map(([nom]) => NOMS[nom] || nom);
    e.innerHTML = d.les_deux
      ? '<span class="pastille oui"></span>deux lecteurs prêts'
      : `<span class="pastille non"></span>${txt(absents.join(' et '))} injoignable`
        + `${absents.length > 1 ? 's' : ''}`;
    if (!d.les_deux) depot.setAttribute('aria-disabled', 'true');
  } catch {
    e.innerHTML = '<span class="pastille non"></span>API injoignable';
  }
})();
