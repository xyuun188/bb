-- PostgreSQL. Roll the application back first. Keep all recorded clock values.
BEGIN;
DROP TRIGGER IF EXISTS bb_history_settlement_clock ON okx_position_history;
DROP FUNCTION IF EXISTS bb_stamp_history_settlement();
COMMIT;
