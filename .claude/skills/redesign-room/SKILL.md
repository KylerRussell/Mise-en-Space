---
name: redesign-room
description: Redesign a scanned room from a text request. Produces 2-3 layout variants in the scene backend plus a shopping list with prices and purchase links. Use when the user asks to restyle, rearrange or furnish a room loaded in Mise-en-Space.
---

# Redesign a room (Stage 6 agent)

You have the `mise` MCP tools for everything spatial and your own web search and fetch
for style research and products. The model proposes poses and relations; the solver and
`check_layout` enforce physics. Units are meters; yaw is degrees; an object's front is its +y.

Never edit `main`. Work only in variants you create. Locked objects never move.

## 1. Brief
Turn the request into a structured brief: style words, budget (ask if none; there is no
default yet), must-keep items, what the room is for, hard constraints. Show it back in one line.

## 2. Style card
Search the web for the style. Write a card: palette, materials, signature furniture types,
density, things to avoid, with source links. Ask the user to confirm or edit it before
any shopping. This is the cheap checkpoint.

## 3. Gap analysis
`get_room`, `list_objects`, `render_view(plan)` and `render_view(eye)`. Decide what to keep,
move, hide and add, by label number and id. If the home has several spaces and the request
is about one ("redo the bedroom"), pass that `space` to these tools and to `solve_layout`,
and leave the other rooms alone. Open-plan zones (living room + kitchen) share a floor, so
check walkways across the zone line too.

## 4. Source products
For each item to add, compute the largest footprint that fits (free floor, walkways
>= 0.9 m, door swings clear). Try `search_products` first (official retailer APIs only),
then your own web search. Drop listings without dimensions. Respect the budget.

## 5. Import
`import_product(url, category)` for each shortlisted item. Drop any whose status is
`fallback_box` unless no alternative exists, and say so. Note `size mismatch` warnings.

## 6. Lay out
`create_variant` (one per concept, 2-3 total). Hide removed items, `add_object` new ones,
then call `solve_layout` with relations, e.g.
`{"kind": "against_wall", "obj": "sofa_1", "target": "2"}`,
`{"kind": "in_front_of", "obj": "coffee_table_1", "target": "sofa_1", "min": 0.35, "max": 0.5}`,
`{"kind": "facing", "obj": "armchair_1", "target": "sofa_1"}`,
`{"kind": "away_from_window", "obj": "bookshelf_1"}`.
Use `move_object` / `place_relative` for single tweaks and fix every violation it reports.

Orientation: unless `upright_required` is true, an object may be tipped in 90° steps
(`move_object(..., roll=90)` stands a wide, low shelf on its end). Listings often show only
one way. For tippable pieces, try both orientations when space or style is tight and keep the
better one; mention the choice in the change list. Never tip an upright-only object.

## 7. Critique
`render_view` plan and eye-level; check against the brief and the style card; revise.
At most 3 rounds per variant. `check_layout` must show zero hard violations.

## 8. Present
For each variant: renders, `diff_variants(main, variant)` as a change list, and
`shopping_list` (item, retailer, price, listed W x D x H, link, date checked, total).
Re-check price and stock with a fresh fetch just before presenting. Tell the user they can
accept changes item by item in the editor's "Proposal vs main" panel.

Default rules (editable in the scene's `rules`): main walkways >= 0.9 m, door swings clear,
nothing tall in front of windows, locked items never move.
