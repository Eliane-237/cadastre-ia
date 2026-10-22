/* Projection spatiale des parcelles enregistrées.
 *
 * La couche vient de GET /parcelles/geojson. Elle est reprojetée en WGS84 sans
 * transformation de datum — l'API le dit dans `note`, et cette page l'affiche :
 * bon pour regarder où se trouve une parcelle, pas pour mesurer un recouvrement.
 * Le chiffre qui compte reste celui de /parcelles/{id}/recouvrements, calculé en
 * UTM 28N sur les coordonnées imprimées.
 */

const API = '';

/* Leaflet est servi depuis le projet. S'il manque quand même, la page le dit
 * plutôt que de rester muette. */
if (typeof L === 'undefined') {
  document.getElementById('carte').innerHTML =
    '<p style="padding:24px;color:#5c666e">La bibliothèque de carte '
    + '(web/vendor/leaflet/leaflet.js) est introuvable : le dossier web/vendor '
    + 'n\'a pas suivi le reste du projet. Remettez-le en place et rechargez.</p>';
  throw new Error('leaflet absent');
}

const ENCRE   = '#1a4d7a';
const CACHET  = '#a8201a';
const TRAIT   = { color: ENCRE, weight: 1.5, fillColor: ENCRE, fillOpacity: 0.08 };
const RETENUE = { color: ENCRE, weight: 2.5, fillOpacity: 0.2 };
const CONFLIT = { color: CACHET, weight: 2, fillColor: CACHET, fillOpacity: 0.25 };

const carte = L.map('carte', { zoomControl: true })
  .setView([14.6937, -17.4441], 11);           // Dakar, le temps que la couche arrive

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '© OpenStreetMap',
  opacity: 0.55,                                // le fond reste un fond, les parcelles dominent
}).addTo(carte);

L.control.scale({ imperial: false }).addTo(carte);

const formes = new Map();                       // document_id -> couche Leaflet
let retenu = null;

/* --------------------------------------------------------------- la couche */

(async function charger() {
  try {
    const r = await fetch(`${API}/parcelles/geojson`);
    if (!r.ok) throw new Error(`Erreur ${r.status}`);
    const fc = await r.json();

    document.getElementById('note').textContent = fc.note || '';
    const traits = fc.features || [];
    document.getElementById('compte').innerHTML =
      `<b>${traits.length}</b> parcelle${traits.length > 1 ? 's' : ''} en base`;
    document.getElementById('registre-vide').hidden = traits.length > 0;

    if (!traits.length) return;

    const couche = L.geoJSON(fc, {
      style: TRAIT,
      onEachFeature: (trait, c) => {
        formes.set(trait.properties.document, c);
        c.on('click', () => selectionner(trait.properties.document));
        c.bindTooltip(etiquette(trait.properties), { sticky: true });
      },
    }).addTo(carte);

    carte.fitBounds(couche.getBounds(), { padding: [40, 40] });
    listerParcelles(traits);
    poserConfrontation();
  } catch (err) {
    document.getElementById('note').textContent = `Couche indisponible : ${err.message}`;
  }
})();

const etiquette = (p) => `${txt(p.titre || p.document)} — `
  + `${Math.round(p.aire_m2).toLocaleString('fr-FR')} m²`;

function listerParcelles(traits) {
  const liste = document.getElementById('liste');
  liste.innerHTML = '';
  for (const t of traits) {
    const p = t.properties;
    const li = document.createElement('li');
    li.innerHTML = `
      <button data-doc="${attr(p.document)}">
        <span class="titre">${txt(p.titre || 'titre non lu')}</span><br>
        <span class="detail">${Math.round(p.aire_m2).toLocaleString('fr-FR')} m²
          · ${txt(p.document)}</span>
        ${p.comparable ? '' : '<br><span class="alerte">datum inconnu, non comparable</span>'}
      </button>`;
    li.querySelector('button').addEventListener('click',
      () => selectionner(p.document, p));
    liste.append(li);
  }
}

/* ------------------------------------------------------- une parcelle, de près */

async function selectionner(doc, props) {
  if (retenu && formes.has(retenu)) formes.get(retenu).setStyle(TRAIT);
  retenu = doc;

  const forme = formes.get(doc);
  if (forme) {
    forme.setStyle(RETENUE);
    carte.fitBounds(forme.getBounds(), { padding: [80, 80] });
    props = props || forme.feature.properties;
  }

  document.querySelectorAll('#liste button').forEach((b) => {
    if (b.dataset.doc === doc) b.setAttribute('aria-current', 'true');
    else b.removeAttribute('aria-current');
  });

  const hote = document.getElementById('detail');
  const ecart = props.aire_imprimee_m2
    ? ((Math.abs(props.aire_m2 - props.aire_imprimee_m2) / props.aire_imprimee_m2) * 100)
      .toFixed(2) + ' %'
    : '—';

  hote.innerHTML = `
    <div class="detail-parcelle">
      <dl>
        <dt>Titre</dt><dd>${txt(props.titre || '—')}</dd>
        <dt>Titre mère</dt><dd>${txt(props.titre_mere || '—')}</dd>
        <dt>Surface calculée</dt><dd>${Math.round(props.aire_m2).toLocaleString('fr-FR')} m²</dd>
        <dt>Surface imprimée</dt><dd>${props.aire_imprimee_m2
          ? props.aire_imprimee_m2.toLocaleString('fr-FR') + ' m²' : '—'}</dd>
        <dt>Écart</dt><dd>${ecart}</dd>
        <dt>Référentiel</dt><dd>${txt(props.datum)}</dd>
      </dl>
      <p class="vide" id="rec" style="margin:8px 0 0">Recherche des voisines…</p>
      <img src="${API}/parcelles/${encodeURI(doc)}/apercu"
           alt="Tracé de la parcelle ${attr(props.titre || doc)}">
    </div>`;

  try {
    const r = await fetch(`${API}/parcelles/${encodeURI(doc)}/recouvrements`);
    const d = await r.json();
    const rec = document.getElementById('rec');
    if (!r.ok) { rec.textContent = d.detail || `Erreur ${r.status}`; return; }

    if (!d.recouvrements.length) {
      rec.textContent = 'Aucune voisine ne recouvre cette parcelle.';
      return;
    }
    rec.outerHTML = `<div class="message ${d.conflits ? 'message--erreur' : ''}"
         style="margin:8px 0 0">
      ${d.recouvrements.length} recouvrement${d.recouvrements.length > 1 ? 's' : ''},
      dont ${d.conflits} à instruire.
      <ul style="margin:4px 0 0;padding-left:1.1em">${d.recouvrements.map((x) =>
        `<li>${txt(x.b)} — ${txt(x.message)}</li>`).join('')}</ul>
    </div>`;

    for (const x of d.recouvrements) {
      if (x.conflit && formes.has(x.b)) formes.get(x.b).setStyle(CONFLIT);
    }
  } catch (err) {
    const rec = document.getElementById('rec');
    if (rec) rec.textContent = `Recouvrements indisponibles : ${err.message}`;
  }
}

/* La dernière confrontation faite depuis la page Lecture : la parcelle en cours
 * d'instruction n'est pas en base, elle se superpose en pointillé. */
function poserConfrontation() {
  const brut = sessionStorage.getItem('cadastre.confrontation');
  if (!brut) return;
  try {
    const { document: doc, geojson } = JSON.parse(brut);
    const seule = (geojson.features || []).filter((f) => f.properties.document === doc);
    if (!seule.length) return;
    const couche = L.geoJSON({ type: 'FeatureCollection', features: seule }, {
      style: { color: CACHET, weight: 2, dashArray: '5 4', fillOpacity: 0 },
    }).addTo(carte);
    couche.bindTooltip('Dernière confrontation, non enregistrée');
  } catch { /* stockage illisible : la carte reste utilisable sans */ }
}

const txt  = (v) => String(v ?? '').replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const attr = (v) => (v === null || v === undefined ? '' : txt(v));
