//===-- CoarrayOps.td - Coarray operation definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the Coarray dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_COARRAY_COARRAY_OPS
#define FORTRAN_DIALECT_COARRAY_COARRAY_OPS

include "flang/Optimizer/Dialect/Coarray/CoarrayDialect.td"
include "flang/Optimizer/Dialect/FIRTypes.td"
include "flang/Optimizer/Dialect/FIRAttr.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"

class coarray_Op<string mnemonic, list<Trait> traits>
    : Op<CoarrayDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Initialization and Finalization
//===----------------------------------------------------------------------===//

def coarray_InitOp : coarray_Op<"init", []> {
  let summary = "Initialize the parallel environment";
  let description = [{This operation will initialize the parallel environment}];

  let arguments = (ins AnyIntegerType:$stat);
}

def coarray_StopOp : coarray_Op<"stop", [AttrSizedOperandSegments]> {
  let summary = "Operation that synchronizes all execution images, clean up the"
                "parallel runtime environment, and terminates the program.";
  let description = [{
    This operation synchronizes all executing images, cleans up the parallel 
    runtime environment, and terminates the program. Calls to this operation do 
    not return. This operation supports both normal termination at the end of a
    program, as well as any STOP statements from the user source code.
  }];

  let arguments = (ins Optional<BoolLike>:$quiet,
      Optional<AnyIntegerType>:$stop_code_int,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$stop_code_char,
      UnitAttr:$is_error);

  let hasVerifier = 1;
}

def coarray_FailImageOp : coarray_Op<"fail_image", [NoMemoryEffect]> {
  let summary = 
    "Causes the executing image to cease participating in program execution";
}

//===----------------------------------------------------------------------===//
// Image Queries
//===----------------------------------------------------------------------===//

def coarray_NumImagesOp
    : coarray_Op<"num_images", [NoMemoryEffect, AttrSizedOperandSegments]> {
  let summary = "Query the number of images in the specified or current team";
  let description = [{
    This operation query the number of images in the specified or current
    team and can be called with 3 differents way :
    - `num_images()`
    - `num_images(team)`
    - `num_images(team_number)`

    Argumemnts:
    - `team` : Shall be a scalar of type `team_type` from the `ISO_FORTRAN_ENV`
            module with a value that identifies the current or ancestor team.
    - `team_number` :  Shall be an integer scalar. It shall identify the
            initial team or a sibling team of the current team.

    Result Value: The number of images in the specified team, or in the current
    team if no team is specified.
  }];

  let arguments = (ins Optional<AnyInteger>:$team_number, 
                       Optional<AnyRefOrBoxType>:$team);
  let results = (outs AnyIntegerType);
  
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"mlir::Value", "{}">:$team_number,
                   CArg<"mlir::Value", "{}">:$team),
    [{ return build($_builder, $_state, team_number, team); }]>
  ];
  
  let hasVerifier = 1;
}

def coarray_ThisImageOp : coarray_Op<"this_image", [AttrSizedOperandSegments]> {
  let summary = "Determine the image index of the current image";
  let description = [{
    Arguments:
    - `coarray` :  Shall be a coarray of any type.
    - `dim` : Shall be an integer scalar. Its value shall be in the range of
          1 <= DIM <= N, where N is the corank of the coarray.
    - `team`(optional) : Shall be a scalar of type `team_type` from 
          ISO_FORTRAN_ENV. If the `coarray` is present, it shall be 
          established in that team.

    Results:
    - Case(1) : The result of `this_image([team])` is a scalar with a value 
          equal to the index of the image in the current or specified team.
    - Case(2) : The result of `this_image(coarray [,team])` is the sequence of 
          cosubscript vales for `coarray`.
    - Case(3) : The result of `this_image(coarray, dim [,team])` is the value of
          cosubscript `dim` in the sequence of cosubscript values for `coarray`.

    Example: 
    ```fortran
      REAL :: A[10, 0:9, 0:*]
    ```
    If we take a look on the example and we are on image 5, `this_image` has the 
    value 5, `this_image(A)` has the value [5, 0, 0].
  }];

  let arguments = (ins Optional<fir_BoxType>:$coarray,
                       Optional<AnyInteger>:$dim,
                       Optional<AnyRefOrBoxType>:$team);
  let results = (outs ArrayOrIntegerType); 

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$coarray, 
                   "int32_t":$dim, 
                    CArg<"mlir::Value", "{}">:$team)>,
    OpBuilder<(ins "mlir::Value":$coarray, 
                   "mlir::Value":$dim, 
                    CArg<"mlir::Value", "{}">:$team)>,
    OpBuilder<(ins "mlir::Value":$coarray, 
                    CArg<"mlir::Value", "{}">:$team)>,
    OpBuilder<(ins  CArg<"mlir::Value", "{}">:$team)>
  ];

  let hasVerifier = 1;
}

def coarray_FailedImagesOp : coarray_Op<"failed_images", []> {
  let summary = "Indices of failed images.";
  let description = [{
    Argument: `team`(optional) : Shall be a scalar of type `team_type` from 
          ISO_FORTRAN_ENV.
    Result: The values of the image indices of the known failed images
      in the specified team, in numerically increasing order.
  }];
  
  let arguments = (ins Optional<AnyRefOrBoxType>:$team);
  let results = (outs fir_SequenceType); 
}

def coarray_StoppedImagesOp : coarray_Op<"stopped_images", []> {
  let summary = "Indices of stopped images.";
  let description = [{
    Argument: `team`(optional) : Shall be a scalar of type `team_type` from 
          ISO_FORTRAN_ENV.
    Result: The values of the image indices of the known stopped images
      in the specified team, in numerically increasing order.
  }];
  
  let arguments = (ins Optional<AnyRefOrBoxType>:$team);
  let results = (outs fir_SequenceType); 
}

def coarray_ImageStatusOp : coarray_Op<"image_status", []> {
  let summary = "Image execution state.";
  let description = [{
    Arguments: 
      - `image`: Its value shall positive and less than or equal to the number
        of images in the specified `team`.
      - `team`(optional) : Shall be a scalar of type `team_type` from 
        ISO_FORTRAN_ENV. If `team` is absent, the team specified is the
        current team.
    Result: The values of the image indices of the known stopped images
      in the specified team, in numerically increasing order.
  }];
  
  let arguments = (ins Optional<AnyRefOrBoxType>:$team);
  let results = (outs fir_SequenceType); 
}

//===----------------------------------------------------------------------===//
// Coarray Queries
//===----------------------------------------------------------------------===//

def coarray_ImageIndexOp
    : coarray_Op<"image_index", [AttrSizedOperandSegments]> {
  let summary = "Image index from cosubscripts.";
  let description = [{
    Arguments:
      - `coarray`: Shall be a coarray of any type.
      - `sub`: rank-one integer array of size equal to the corank of `coarray`.
        Shall be a valid sequence of cosubscripts for `coarray` in the team 
        specified by `team` or `team_number`, or in the current team.
      - `team`: Shall be a scalar of type `team_type` from ISO_FORTRAN_ENV.
      - `team_number`: It shall identify the initial team or a sibling team 
        of the current team.
    
    Usage:
    - Case(1) : `call image_index(coarray, sub)`
    - Case(2) : `call image_index(coarray, sub, team)`
    - Case(3) : `call image_index(coarray, sub, team_number)`
    
    Result: Return an integer scalar which represent the image index from `sub`
      for `coarray`.
  }];

  let arguments = (ins fir_BoxType:$coarray,
                       fir_BoxType:$sub, 
                       Optional<AnyRefOrBoxType>:$team,
                       Optional<AnyInteger>:$team_number);

  let results = (outs AnyIntegerType);
}

def coarray_LcoboundOp : coarray_Op<"lcobound", [NoMemoryEffect]> {
  let summary = "Returns the lower cobound(s) associated with a coarray.";
  let description = [{
    This operation returns the lower cobound(s) associated with a coarray.
    Arguments: 
      - `coarray`: Shall be a coarray of any type.
      - `dim` : Shall be an integer scalar. Its value shall be in the range of
          `1 <= DIM <= N`, where `N` is the corank of the coarray.
    Results:
      - Case(1): If `dim` is present, the result is an integer scalar equal to
        the lower cobound for codimension `dim`.
      - Case(2): `dim` is absent, so the result is an array whose size matches
        the corank of the indicated coarray.
  }];

  let arguments = (ins fir_BoxType:$coarray);
  let results = (outs fir_SequenceType); 
}

def coarray_UcoboundOp : coarray_Op<"ucobound", [NoMemoryEffect]> {
  let summary = "Returns the upper cobound(s) associated with a coarray.";
  let description = [{
    This operation returns the upper cobound(s) associated with a coarray.
    Arguments: 
      - `coarray`: Shall be a coarray of any type.
      - `dim` : Shall be an integer scalar. Its value shall be in the range of
          `1 <= DIM <= N`, where `N` is the corank of the coarray.
    Results:
      - Case(1): If `dim` is present, the result is an integer scalar equal to
        the upper cobound for codimension `dim`.
      - Case(2): `dim` is absent, so the result is an array whose size matches
        the corank of the indicated coarray.
  }];

  let arguments = (ins fir_BoxType:$coarray);
  let results = (outs fir_SequenceType); 
}

def coarray_CoshapeOp : coarray_Op<"coshape", [NoMemoryEffect]> {
  let summary = "Return the sizes of codimensions of a coarray.";
  let description = [{
    Argument: `coarray`: Shall be a coarray of any type.
    Result : Is an array whose size matches the corank of the indicated coarray and
       returns `UCOBOUND - LCOBOUND + 1`.
  }];

  let arguments = (ins fir_BoxType:$coarray);
  let results = (outs fir_SequenceType); 
}

//===----------------------------------------------------------------------===//
// Allocation and Deallocation
//===----------------------------------------------------------------------===//

def coarray_AllocateOp
    : coarray_Op<"allocate", [MemoryEffects<[MemAlloc<DefaultResource>]>]> {
  let summary = "Perform the allocation of a coarray and provide a "
                "corresponding coarray descriptor";

  let description = [{
    This operation allocates a coarray and provides a handle referencing a
    corresponding coarray descriptor. This call is collective over the
    current team.
  }];

  let arguments = (ins fir_SequenceType:$lcobounds, 
                       fir_SequenceType:$ucobound,
                       I64:$size_in_bytes, 
                       FuncType:$final_func,
                       Arg<AnyRefOrBoxType, "", [MemWrite]>:$coarray_desc,
                       Arg<AnyRefOrBoxType, "", [MemWrite]>:$allocated_memory,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let results = (outs AnyIntegerType:$stat);

  let hasVerifier = 1;
}

def coarray_DeallocateOp
    : coarray_Op<"deallocate", [MemoryEffects<[MemFree<DefaultResource>]>]> {
  let summary = "Perform the deallocation of a coarray";
  let description = [{
    The coarray.deallocate will releases memory allocated of an allocatable
    coarray.
  }];
  let arguments = (ins Arg<AnyRefOrBoxType, "", [MemRead]>:$coarray,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);

  let hasVerifier = 1;
}

def coarray_AliasCreateOp : coarray_Op<"alias_create", []> {
  let summary = "Create a new coarray descriptor aliased.";
  
  let arguments = (ins AnyRefOrBoxType:$coarray_desc,
                       fir_SequenceType:$lcobounds, 
                       fir_SequenceType:$ucobound);
  let results = (outs fir_BoxType);
}

def coarray_AliasDestroyOp : coarray_Op<"alias_destroy", []> {
  let summary = "Delete an alias descriptor.";
  
  let arguments = (ins AnyRefOrBoxType:$coarray_desc,
                       fir_SequenceType:$lcobounds, 
                       fir_SequenceType:$ucobound);
  let results = (outs fir_BoxType);
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def coarray_SyncAllOp : coarray_Op<"sync_all", []> {
  let summary =
      "Performs a collective synchronization of all images in the current team";

  let arguments = (ins Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_SyncImagesOp : coarray_Op<"sync_images", []> {
  let summary =
      "Performs a synchronization of image with each of the other images in the `image_set`";

  let arguments = (ins fir_SequenceType:$image_set,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
  let hasVerifier = 1;
}

def coarray_SyncMemoryOp : coarray_Op<"sync_memory", []> {
  let summary = "Operation that ends one segment and begins another.";
  let description = [{
    Operation that ends one segment and begins another; Those two segments can 
    be ordered by user-defined way with respect to segments on other images.
  }];

  let arguments = (ins Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_SyncTeamOp : coarray_Op<"sync_team", []> {
  let summary = "Performs a synchronization of the team, identified by `team_value`";

  let arguments = (ins Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

//===----------------------------------------------------------------------===//
// Locks 
//===----------------------------------------------------------------------===//

def coarray_LockOp : coarray_Op<"lock", [AttrSizedOperandSegments]> {
  let summary =
      "Wait until an identified lock variable is unlocked and then locks.";
  let description = [{
    A lock variable is defined atomically by a LOCK or UNLOCK statement.
    This opearation on multiple images attempt to acquire a lock, one will
    succeed and the others will either fail if `acquired_lock` is present, 
    or will wait until the lock is later released if `acquired_lock` does not
    appear.
    
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        the lock variable to be locked.
      - `lock_var` : Shall be a coarray of type LOCK_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
      - `acquired_lock (optional)` : Shall be a logical scalar variable.
  }];

  let arguments = (ins AnyIntegerType:$image_num, fir_BoxType:$lock_var,
      Optional<fir_BoxType>:$acquired_lock,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_UnlockOp : coarray_Op<"unlock", []> {
  let summary = "The lock variable become unlocked.";
  let description = [{
    A lock variable is defined atomically by a LOCK or UNLOCK statement.
    This operation, for an identified lock variable, will unlock the the variable.
    
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        the lock variable to be locked.
      - `lock_var` : Shall be a coarray of type LOCK_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num, fir_BoxType:$lock_var,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

//===----------------------------------------------------------------------===//
// Critical 
//===----------------------------------------------------------------------===//

def coarray_CriticalOp : coarray_Op<"critical", []> {
  let summary = "Reprensent the CRITICAL statement on a specific coarray.";
  let description = [{
    Fortran provide CRITICAL construct to limits execution of block to one
    image at time. The compiler shall define a coarray and establish it in
    the initial team.
  }];

  let arguments = (ins fir_BoxType:$coarray,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_EndCriticalOp : coarray_Op<"end_critical", []> {
  let summary = "Reprensent the END CRITICAL statement on a specific coarray.";
  let description = [{
    Fortran provide CRITICAL construct to limits execution of block to one
    image at time. The compiler shall define a coarray and establish it in
    the initial team.
    END CRITICAL statement will completes execution of the CRITICAL construct
    associated to `coarray`.
  }];

  let arguments = (ins fir_BoxType:$coarray,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

//===----------------------------------------------------------------------===//
// Events and Notifications 
//===----------------------------------------------------------------------===//

def coarray_EventPostOp : coarray_Op<"event_post", []> {
  let summary = "Posts an event.";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        the event variable.
      - `event_var`: Shall be a coarray of type EVENT_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num, fir_BoxType:$event_var,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_EventWaitOp : coarray_Op<"event_wait", [AttrSizedOperandSegments]> {
  let summary = "Waits until an event is posted.";
  let description = [{
    Arguments: 
      - `event_var`: Shall be a coarray of type EVENT_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
      - `until_count` : Shall be an integer scalar and indicate the count of 
        the given event variable to be waited for.
  }];

  let arguments = (ins fir_BoxType:$event_var,
      Optional<AnyIntegerType>:$until_count,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_EventQueryOp : coarray_Op<"event_query", []> {
  let summary = "Query the count of an event on the calling image.";
  let description = [{
    Argument: `event_var` shall be a coarray of type EVENT_TYPE from the intrinsic
      module ISO_FORTRAN_ENV.
    Result: `count` shall be an integer scalar and indicate the current count 
      of event variables.
  }];

  let arguments = (ins fir_BoxType:$event_var);
  let results = (outs AnyIntegerType:$count, AnyIntegerType:$stat);
}

def coarray_NotifyWaitOp
    : coarray_Op<"notify_wait", [AttrSizedOperandSegments]> {
  let summary = "Wait on notification of an incoming put operation.";
  let description = [{
    Arguments: 
      - `event_var`: Shall be a coarray of type EVENT_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
      - `until_count` : Shall be an integer scalar and indicate the count of 
        the given event variable to be waited for.
  }];

  let arguments = (ins fir_BoxType:$event_var,
      Optional<AnyIntegerType>:$until_count,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

//===----------------------------------------------------------------------===//
// Teams
//===----------------------------------------------------------------------===//

def coarray_FormTeamOp : coarray_Op<"form_team", [AttrSizedOperandSegments]> {
  let summary =
      "Create a set of sibling teams whose parent team is the current team.";
  let description = [{
    Create a new team for each unique `team_number` value specified.
    Each executing image will belong to the team whose `team_number` is equal 
    to the value of team-number on that image, and `team_var` becomes defined
    with a value that identifies that team.

    If `new_index` is specified, the image index of the executing image will take
    this index in its new team. Otherwise, the new image index is processor 
    dependent.
    
    Arguments: 
      - `team_number`: Shall be a positive integer.
      - `team_var` : Shall be a coarray of type TEAM_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
      - `new_index`: Shall be an integer that correspond to the index that
        the calling image will have in the new team.
  }];

  let arguments = (ins AnyIntegerType:$team_number,
      Arg<fir_BoxType, "", [MemWrite]>:$team_var,
      Optional<AnyIntegerType>:$new_index,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_ChangeTeamOp : coarray_Op<"change_team", []> {
  let summary = "Changes the current team.";
  let description = [{
    Fortran provide CHANGE TEAM construct to changes the current team and
    CHAND TEAM operation will change the current team to the specified new
    team. 
  }];

  let arguments = (ins fir_BoxType:$team,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_EndTeamOp : coarray_Op<"end_team", []> {
  let summary = "Changes the current team to the parent team.";
  let description = [{
    Fortran provide CHANGE TEAM construct to changes the current team.
    Executing END TEAM operation restores the current team to the original team
    that was current for the CHANGE TEAM operation.
  }];

  let arguments = (ins Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_GetTeamOp : coarray_Op<"get_team", []> {
  let summary = "Get the team value for the current or ancestor team.";
  let description = [{
    This operation get the team value for the current or ancestor team.
    If `level` isn't present or has the value `CURRENT_TEAM` the returned
    value is the current team.
    `level` cane take one of the following constants : `INITIAL_TEAM`, 
    `PARENT_TEAM` or `CURRENT_TEAM` from the module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins Optional<AnyIntegerType>:$level);
  let results = (outs fir_BoxType:$team);
}

def coarray_TeamNumberOp : coarray_Op<"team_number", []> {
  let summary = "Get the team number";
  let description = [{
    Argument: `team` is optional and shall be a scalar of type TEAM_TYPE from 
    module ISO_FORTRAN_ENV and the vlaue idenfies the current or an ancestor team. 
    If `team` is absent, the team specified is the current team.
  }];

  let arguments = (ins Optional<fir_BoxType>:$team);
  let results = (outs AnyIntegerType);
}

//===----------------------------------------------------------------------===//
// Atomic Operations 
//===----------------------------------------------------------------------===//

def coarray_AtomicAddOp : coarray_Op<"atomic_add", []> {
  let summary = "Atomic addition";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom` and `value`: Shall be a coarray of type integer with kind ATOMIC_INT_KIND 
        from module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemRead]>:$value,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_AtomicAndOp : coarray_Op<"atomic_and", []> {
  let summary = "Atomic bitwise AND";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom` and `value`: Shall be a coarray of type integer with kind ATOMIC_INT_KIND 
        from module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemRead]>:$value,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_AtomicOrOp : coarray_Op<"atomic_or", []> {
  let summary = "Atomic bitwise OR";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom` and `value`: Shall be a coarray of type integer with kind ATOMIC_INT_KIND 
        from module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemRead]>:$value,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_AtomicXorOp : coarray_Op<"atomic_xor", []> {
  let summary = "Atomic bitwise XOR";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom` and `value`: Shall be a coarray of type integer with kind ATOMIC_INT_KIND 
        from module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemRead]>:$value,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_AtomicCasOp : coarray_Op<"atomic_cas", []> {
  let summary = "Atomic compare and swap";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom`, `old`, `new` and `compared`: Shall be a coarray of type integer with kind
         ATOMIC_INT_KIND or ATOMIC_LOGICAL_KIND from module ISO_FORTRAN_ENV. They need to 
         have the same type and kind.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemWrite]>:$old,
                       Arg<fir_BoxType, "", [MemRead]>:$compare,
                       Arg<fir_BoxType, "", [MemRead]>:$new_val,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_AtomicDefineOp : coarray_Op<"atomic_define", []> {
  let summary = "Define a variable atomically";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom` and `value`: Shall be a coarray of type integer with kind ATOMIC_INT_KIND 
        or ATOMIC_LOGICAL_KIND from module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemRead]>:$value,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_AtomicRefOp : coarray_Op<"atomic_ref", []> {
  let summary = "Reference a variable atomically";
  let description = [{
    Arguments: 
      - `image_num`: Shall be an integer that correspond to the image index of 
        `atom` where the operation need to be performed. 
      - `atom` and `value`: Shall be a coarray of type integer with kind ATOMIC_INT_KIND 
        or ATOMIC_LOGICAL_KIND from module ISO_FORTRAN_ENV.
  }];

  let arguments = (ins AnyIntegerType:$image_num,
                       fir_BoxType:$atom,
                       Arg<fir_BoxType, "", [MemRead]>:$value,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

//===----------------------------------------------------------------------===//
// Collective Operations
//===----------------------------------------------------------------------===//

def coarray_CoBroadcastOp : coarray_Op<"co_broadcast", [AttrSizedOperandSegments]> {
  let summary = "Broadcast value to images.";
  let description = [{
    The coarray.co_broadcast operation performs the computation of the sum
    across images.
  }];

  let arguments = (ins fir_BoxType:$a,
                       Optional<AnyIntegerType>:$source_image,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_CoMaxOp : coarray_Op<"co_max", [AttrSizedOperandSegments]> {
  let summary = "Compute maximum value across images.";
  let description = [{
    The coarray.co_max operation performs the computation of the maximum 
    across images.
  }];

  let arguments = (ins fir_BoxType:$a,
                       Optional<AnyIntegerType>:$result_image,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_CoMinOp : coarray_Op<"co_min", [AttrSizedOperandSegments]> {
  let summary = "Compute minimum value across images.";
  let description = [{
    The coarray.co_min operation performs the computation of the minimum
    across images.
  }];

  let arguments = (ins fir_BoxType:$a,
                       Optional<AnyIntegerType>:$result_image,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_CoReduceOp : coarray_Op<"co_reduce", [AttrSizedOperandSegments]> {
  let summary = "Generalized reduction across images.";
  let description = [{
    The coarray.co_sum operation performs the reduction across images.
    The `operation` argument shall be a function with exactly two arguments.
    Each arguments and the result shall be scalar, nonallocatable, noncoarray, 
    nonpointer, nonpolymorphic data and with the same type and kind as the 
    argument `a` of the operation coarray.co_reduce.
  }];

  let arguments = (ins fir_BoxType:$a,
                       FuncType:$operation_arg,
                       Optional<AnyIntegerType>:$result_image,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_CoSumOp : coarray_Op<"co_sum", [AttrSizedOperandSegments]> {
  let summary = "Compute sum across images.";
  let description = [{
    The coarray.co_sum operation performs the computation of the sum
    across images.
  }];

  let arguments = (ins fir_BoxType:$a,
                       Optional<AnyIntegerType>:$result_image,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

//===----------------------------------------------------------------------===//
// Coarray Access Operations
//===----------------------------------------------------------------------===//

def coarray_GetOp : coarray_Op<"get", []> {
  let summary = "Fetch data in a coarray from an image number";
  
  let arguments = (ins AnyIntegerType:$image_num,
                       Arg<fir_BoxType, "", [MemRead]>:$coarray,
                       I64:$offset,
                       I64:$size_in_bytes,
                       Arg<AnyRefOrBoxType, "", [MemWrite]>:$dest,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

def coarray_PutOp : coarray_Op<"put", []> {
  let summary = "Assign data to a coarray to an image number";
  
  let arguments = (ins AnyIntegerType:$image_num,
                       Arg<fir_BoxType, "", [MemWrite]>:$coarray,
                       I64:$offset,
                       I64:$size_in_bytes,
                       Arg<AnyRefOrBoxType, "", [MemRead]>:$src,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let results = (outs AnyIntegerType:$stat);
}

#endif // FORTRAN_DIALECT_COARRAY_COARRAY_OPS
