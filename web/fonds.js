/* Les fonds de carte, définis une seule fois.
 *
 * Les deux pages en ont besoin : la Lecture pour situer la parcelle qu'on vient de
 * lire, la Carte pour le registre. Dupliquer deux URL de tuiles et leurs mentions
 * légales dans deux fichiers, c'est s'assurer qu'elles finiront par diverger.
 */

const ENCRE  = '#1a4d7a';
const CACHET = '#a8201a';

/* Un trait bleu fin disparaît sur une image aérienne : sur satellite, les parcelles
 * prennent un liseré plus épais. */
const HALO = { weight: 3, fillOpacity: 0.15 };

function fonds() {
  return {
    Plan: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap',
      opacity: 0.55,            // le fond reste un fond, les parcelles dominent
    }),
    // Imagerie sans clé ni compte, celle qui sert de fond aérien dans QGIS.
    Satellite: L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
      + '/tile/{z}/{y}/{x}',
      { maxZoom: 19, attribution: 'Imagerie © Esri, Maxar, Earthstar Geographics' }),
  };
}

/* Une carte prête : fonds, sélecteur, échelle. Le fond « Plan » est posé d'office. */
function carteNue(element) {
  const carte = L.map(element, { zoomControl: true }).setView([14.6937, -17.4441], 11);
  const couches = fonds();
  couches.Plan.addTo(carte);
  L.control.layers(couches, null, { collapsed: true }).addTo(carte);
  L.control.scale({ imperial: false }).addTo(carte);
  return carte;
}
